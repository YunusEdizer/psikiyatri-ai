#!/usr/bin/env python3
import sys
from psikiyatri_ai.api.main import app

print("=== App Routes Debug ===")
print(f"Total routes: {len(app.routes)}")
print("\nDirect paths:")
for i, route in enumerate(app.routes):
    if hasattr(route, 'path'):
        print(f"  {i}: {route.path} ({route.methods if hasattr(route, 'methods') else 'N/A'})")
    else:
        print(f"  {i}: {type(route).__name__}")

print("\nIncluded router routes:")
for i, route in enumerate(app.routes):
    if hasattr(route, 'original_router'):
        print(f"  IncludedRouter {i} (prefix='{route.include_context.prefix}'):")
        for subroute in route.original_router.routes:
            if hasattr(subroute, 'path'):
                print(f"    - {subroute.path}")
            
# Check OpenAPI
from fastapi.openapi.utils import get_openapi
spec = get_openapi(
    title=app.title,
    version=app.version,
    description=app.description,
    routes=app.routes,
    servers=app.servers,
    tags=app.openapi_tags,
    terms_of_service=app.terms_of_service,
    contact=app.contact,
    license_info=app.license_info,
)

print(f"\nOpenAPI paths: {list(spec['paths'].keys())}")
