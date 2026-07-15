"""DeployAI Frontend Runner Script."""

import sys
from frontend.app import DeployAIApplication

if __name__ == "__main__":
    app = DeployAIApplication()
    sys.exit(app.run())
