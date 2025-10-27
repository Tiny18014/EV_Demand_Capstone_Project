from datetime import datetime
import requests, numpy as np
#from model.agent_charging import chargingmodel_agent
#98e8f848-da3c-4e72-8d11-88334312e4c9
import requests, numpy as np

def fetch_ocm_metrics_india():
    url = "https://api.openchargemap.io/v3/poi/"
    headers = {
        "X-API-Key": "98e8f848-da3c-4e72-8d11-88334312e4c9",  # Replace with your actual key
    }
    params = {
        "output": "json",
        "countrycode": "IN",
        "maxresults": 5000
    }

    r = requests.get(url, headers=headers, params=params)
    if r.status_code != 200:
        raise RuntimeError(f"OCM API error {r.status_code}: {r.text[:200]}")

    try:
        data = r.json()
    except Exception as e:
        print("Raw response:", r.text[:300])
        raise e

    if not data:
        raise ValueError("No data returned from OCM API.")

    # Extract power values safely
    power_values = []
    fast_count = 0

    for x in data:
        if not x.get("Connections"):
            continue
        power_kw = x["Connections"][0].get("PowerKW")
        if power_kw is None:
            continue
        power_values.append(power_kw)
        if power_kw > 22:
            fast_count += 1

    total_chargers = len(data)
    fast_charger_share = fast_count / total_chargers if total_chargers else 0
    avg_power_kw = np.mean(power_values) if power_values else 0

    return {
        "region_name": "India",
        "total_chargers": total_chargers,
        "fast_charger_share": round(fast_charger_share, 2),
        "avg_power_kw": round(avg_power_kw, 1),
    }


live_metrics = fetch_ocm_metrics_india()
#response = chargingmodel_agent.run(live_metrics)
print(live_metrics)
