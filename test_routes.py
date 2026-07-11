#!/usr/bin/env python3
from psikiyatri_ai.api.main import app

paths = set()
for route in app.routes:
    if hasattr(route, 'path'):
        paths.add(route.path)
    elif hasattr(route, 'original_router'):
        for subroute in route.original_router.routes:
            if hasattr(subroute, 'path'):
                paths.add(subroute.path)

print("Registered paths:")
for p in sorted(paths):
    print(f"  {p}")
