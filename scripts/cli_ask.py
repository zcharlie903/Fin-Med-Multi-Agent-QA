#!/usr/bin/env python3
import sys, json, httpx

def main():
    q = sys.argv[1] if len(sys.argv) > 1 else "What lab markers indicate anemia?"
    payload = {"question": q, "domain": "medical", "session_id": "cli"}
    r = httpx.post("http://localhost:8000/ask", json=payload, timeout=60.0)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))

if __name__ == "__main__":
    main()
