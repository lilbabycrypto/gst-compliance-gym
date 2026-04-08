try:
    from ..models import CallToolAction, CallToolObservation
    from .gst_environment import GSTComplianceEnvironment
except (ImportError, ModuleNotFoundError):
    from models import CallToolAction, CallToolObservation
    from server.gst_environment import GSTComplianceEnvironment

from openenv.core.env_server.http_server import create_app

app = create_app(
    GSTComplianceEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="gst_compliance_gym",
)


@app.get("/")
def root():
    return {
        "name": "GST Compliance Gym",
        "description": "OpenEnv environment for Indian GST compliance auditing",
        "endpoints": ["/health", "/reset", "/step", "/state", "/schema", "/metadata"],
    }


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
