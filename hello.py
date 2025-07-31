"""
Entry point for 12-factor agents application.
This file provides backwards compatibility while the main application is in src/
"""

def main():
    print("🤖 12-Factor Agents Application")
    print("For full functionality, use: uv run python -m src.main --help")
    print("Quick start: uv run python -m src.main serve")


if __name__ == "__main__":
    main()
