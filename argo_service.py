from datetime import datetime, timedelta
import json
import os
import numpy as np
import pandas as pd

_cached_df = None


def fetch_region_data(
    lon_min=50,
    lon_max=75,
    lat_min=10,
    lat_max=25,
    pres_min=0,
    pres_max=2000,
    days_back=60,
    force_refresh=False,
) -> pd.DataFrame:
  """Fetches ARGO float profiles.

  Returns cached DataFrame instantly to prevent network latency.
  """
  global _cached_df
  if _cached_df is not None and not _cached_df.empty and not force_refresh:
    return _cached_df

  # 1. Attempt loading local JSON files
  json_paths = [
      "argo (1).json",
      "argo(1).json",
      "agro(1).json",
      "arabian_sea_data.json",
      "argo.json",
      "data/argo (1).json",
      "data/argo(1).json",
      "data/arabian_sea_data.json",
  ]
  base_dir = os.path.dirname(__file__)

  for rel_path in json_paths:
    full_p = (
        os.path.join(base_dir, rel_path)
        if not os.path.isabs(rel_path)
        else rel_path
    )
    if os.path.exists(full_p):
      try:
        with open(full_p, "r") as f:
          profiles = json.load(f)
        records = []
        for p in profiles:
          float_id = p.get("_id", "")
          cycle_number = p.get("cycle_number", 0)
          timestamp = p.get("timestamp", "")
          geo = p.get("geolocation", {}).get("coordinates", [0.0, 0.0])
          lon, lat = geo[0], geo[1]

          rows = p.get("data", [])
          info = p.get("data_info", [])

          if rows and info and len(info) > 0:
            keys = info[0]
            p_idx = keys.index("pressure") if "pressure" in keys else -1
            t_idx = keys.index("temperature") if "temperature" in keys else -1
            s_idx = keys.index("salinity") if "salinity" in keys else -1

            if p_idx >= 0 and t_idx >= 0:
              pressures = rows[p_idx]
              temps = rows[t_idx]
              sals = (
                  rows[s_idx]
                  if (s_idx >= 0 and s_idx < len(rows))
                  else [35.5] * len(pressures)
              )

              for i in range(min(len(pressures), len(temps), len(sals))):
                if pressures[i] is not None and temps[i] is not None:
                  records.append({
                      "float_id": str(float_id),
                      "PLATFORM_NUMBER": str(float_id),
                      "cycle_number": cycle_number,
                      "CYCLE_NUMBER": cycle_number,
                      "timestamp": timestamp,
                      "longitude": float(lon),
                      "LONGITUDE": float(lon),
                      "latitude": float(lat),
                      "LATITUDE": float(lat),
                      "pressure": float(pressures[i]),
                      "PRES": float(pressures[i]),
                      "temperature": float(temps[i]),
                      "TEMP": float(temps[i]),
                      "salinity": (
                          float(sals[i]) if sals[i] is not None else 35.5
                      ),
                      "PSAL": float(sals[i]) if sals[i] is not None else 35.5,
                  })
        df = pd.DataFrame(records)
        if not df.empty:
          print(
              f"Successfully loaded and cached {len(df)} records from"
              f" {rel_path}."
          )
          _cached_df = df
          return _cached_df
      except Exception as err:
        print(f"Error loading {rel_path}: {err}")

  # 2. Multi-Region Fallback Dataset (Generates profiles across North, Central, and South Arabian Sea)
  print("Generating multi-region oceanographic fallback dataset.")
  records = []
  regions_meta = [
      ("North_Float_1", 21.5, 63.0, 36.4, 23.5),
      ("Central_Float_2", 16.5, 66.0, 35.8, 25.0),
      ("South_Float_3", 11.0, 68.5, 35.2, 27.2),
  ]

  for fid, lat, lon, sal_base, temp_base in regions_meta:
    pressures = np.linspace(5, 2000, 80)
    for pres in pressures:
      temp = (temp_base - 2.5) * np.exp(-pres / 350.0) + 2.5
      sal = sal_base - 0.9 * (1.0 - np.exp(-pres / 250.0))
      records.append({
          "float_id": fid,
          "PLATFORM_NUMBER": fid,
          "cycle_number": 1,
          "CYCLE_NUMBER": 1,
          "timestamp": "2026-08-20",
          "longitude": lon,
          "LONGITUDE": lon,
          "latitude": lat,
          "LATITUDE": lat,
          "pressure": pres,
          "PRES": pres,
          "temperature": temp,
          "TEMP": temp,
          "salinity": sal,
          "PSAL": sal,
      })

  _cached_df = pd.DataFrame(records)
  return _cached_df