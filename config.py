import sys
import yaml


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)

    errors = []

    api = cfg.get("api", {})
    for key in ("mistral_key", "telegram_key"):
        val = api.get(key, "")
        if not val or "your-" in val:
            errors.append(f"api.{key} is not configured")

    for section in ("embedder", "explainer", "profile_builder", "reranker"):
        timeout = cfg.get(section, {}).get("timeout", 0)
        if not isinstance(timeout, (int, float)) or timeout <= 0:
            errors.append(f"{section}.timeout is missing or invalid")

    min_interval = cfg.get("rate_limiter", {}).get("min_interval", 0)
    if not isinstance(min_interval, (int, float)) or min_interval <= 0:
        errors.append("rate_limiter.min_interval is missing or invalid")

    if errors:
        print("Config validation failed:")
        for e in errors:
            print(f"  - {e}")
        print(f"\nEdit {path} and try again.")
        sys.exit(1)

    return cfg
