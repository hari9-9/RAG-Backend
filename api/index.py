from app.main import app
from mangum import Mangum

# Wrap FastAPI with Mangum to make it serverless
handler = Mangum(app)
