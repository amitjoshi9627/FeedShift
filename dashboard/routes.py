"""
FeedShift Dashboard Routes - Pure Python OAuth Handling
"""

import logging

import reflex as rx

from dashboard.constants import ColorScheme
from dashboard.state import FeedShiftState
from dashboard.ui import login_button, main_content, sidebar

logger = logging.getLogger(__name__)


@rx.page(route="/auth/callback", on_load=FeedShiftState.process_oauth_from_url)
def auth_callback() -> rx.Component:
    """
    OAuth callback route - Pure Python processing with on_load.

    When this page loads, it automatically triggers the
    process_oauth_from_url event to handle OAuth parameters.
    """
    logger.info("OAuth callback page loading")

    return rx.center(
        rx.vstack(
            rx.spinner(size="3", color=ColorScheme.PURPLE_9),
            rx.text("Processing Reddit authentication...", font_size="1.2em", color=ColorScheme.GRAY_12),
            rx.text("Please wait while we complete your login", font_size="0.9em", color=ColorScheme.GRAY_11),
            spacing="3",
            align="center",
        ),
        height="100vh",
        background=f"radial-gradient(ellipse at center, {ColorScheme.PURPLE_3}, transparent)",
    )


@rx.page(route="/")
def index() -> rx.Component:
    """
    Main application index page - Clean, no JavaScript needed.
    """
    logger.info("Creating main dashboard index page")

    return rx.box(
        # Top-right controls area
        rx.hstack(
            login_button(),
            rx.color_mode.button(size="3"),
            spacing="2",
            align="center",
            position="fixed",
            top="1.25rem",
            right="1.25rem",
            z_index="999",
        ),
        # Show error message if login failed
        rx.cond(
            FeedShiftState.login_error != "",
            rx.box(
                rx.text(FeedShiftState.login_error, color="red", font_size="0.9em"),
                position="fixed",
                top="4rem",
                right="1.25rem",
                z_index="998",
                background="white",
                padding="0.5rem",
                border_radius="4px",
                border="1px solid red",
            ),
        ),
        # Fixed sidebar with all controls
        sidebar(),
        # Main content area
        main_content(),
        # Themed background
        background=f"radial-gradient(ellipse at top, {ColorScheme.PURPLE_3}, transparent), "
        + f"radial-gradient(ellipse at bottom, {ColorScheme.PURPLE_2}, transparent)",
        min_height="100vh",
        width="100%",
        padding_top="2rem",
    )
