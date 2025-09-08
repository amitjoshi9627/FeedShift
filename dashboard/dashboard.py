"""
FeedShift Dashboard Application

Main application configuration and initialization for the FeedShift dashboard.
Registers routes, configures themes, and starts the application server.
"""

import logging

import reflex as rx

from dashboard.constants import ColorScheme
from dashboard.routes import auth_callback, index

logger = logging.getLogger(__name__)


def create_app() -> rx.App:
    """
    Create and configure the FeedShift dashboard application.

    Returns:
        Configured Reflex application instance
    """
    logger.info("Creating FeedShift dashboard application")

    # Configure Reflex application with theme and settings
    app = rx.App(
        theme=rx.theme(
            appearance="dark",
            has_background=True,
            radius="large",
            accent_color=ColorScheme.PRIMARY,
        )
    )

    # Register all pages
    app.add_page(index)
    app.add_page(auth_callback)

    logger.info("FeedShift dashboard application configured and ready to serve")
    return app


# Configure Reflex application with theme and settings
app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        radius="large",
        accent_color=ColorScheme.PRIMARY,
    )
)

# Add the main page to the application
app.add_page(index)

logger.info("FeedShift dashboard application configured and ready to serve")
