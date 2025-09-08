"""
FeedShift Dashboard UI Components

Contains all stateless UI components used throughout the dashboard.
Components are organized by functionality and maintain consistent styling.
"""

import logging
from typing import Any, Dict

import reflex as rx

from dashboard.constants import (
    ColorScheme,
    ContentFields,
    DefaultValues,
    PostSimilarityType,
    UIConstants,
)
from dashboard.state import FeedShiftState
from src.data.constants import DataCols, RedditDataCols

logger = logging.getLogger(__name__)


# === Authentication Components ===


def login_button() -> rx.Component:
    """Enhanced login/logout component with proper state handling."""
    return rx.cond(
        FeedShiftState.is_authenticated,
        rx.hstack(
            rx.avatar(
                fallback=rx.cond(FeedShiftState.reddit_username != "", FeedShiftState.reddit_username[0].upper(), "U"),
                size="2",
                color_scheme=ColorScheme.PRIMARY,
            ),
            rx.vstack(
                rx.text(
                    f"u/{FeedShiftState.reddit_username}",
                    font_weight="600",
                    font_size="0.9em",
                    color=ColorScheme.GRAY_12,
                ),
                rx.text("Connected", font_size="0.8em", color=ColorScheme.GREEN_9, font_weight="500"),
                spacing="0",
                align="start",
            ),
            rx.button(
                rx.hstack(rx.icon("log-out", size=12), rx.text("Logout"), spacing="1"),
                on_click=FeedShiftState.logout,
                variant="solid",
                color_scheme=ColorScheme.SECONDARY,
                size="1",
            ),
            spacing="2",
            align="center",
            padding="0.5rem 0.75rem",
            border_radius="8px",
            background=ColorScheme.AUTH_SUCCESS_BG,
            border=f"1px solid {ColorScheme.AUTH_SUCCESS_BORDER}",
        ),
        rx.button(
            rx.cond(
                FeedShiftState.is_logging_in,
                rx.hstack(
                    rx.spinner(size="1"), rx.text("Connecting...", font_size="0.9em"), spacing="2", align="center"
                ),
                rx.hstack(
                    rx.icon("log-in", size=14),  # ✅ Valid icon
                    rx.text("Login with Reddit", font_size="0.9em"),
                    spacing="2",
                    align="center",
                ),
            ),
            on_click=FeedShiftState.initiate_reddit_login,
            disabled=FeedShiftState.is_logging_in,
            color_scheme=ColorScheme.PRIMARY,
            size="2",
        ),
    )


# === Sidebar Components ===


def subreddit_selector() -> rx.Component:
    """Enhanced subreddit selector component with grid layout."""
    logger.debug("Creating subreddit selector component")

    return rx.card(
        rx.vstack(
            # Header section
            rx.hstack(
                rx.icon("circle-check", size=20, color=ColorScheme.PURPLE_9),
                rx.text(
                    "Select Subreddit",
                    font_size="1.15em",
                    font_weight="600",
                    color=ColorScheme.GRAY_12,
                ),
                align="center",
                spacing="2",
                margin_bottom="0.5rem",
            ),
            # Current selection display
            rx.text(
                "Selected: "
                + f"{rx.cond(FeedShiftState.selected_subreddit, f'r/{FeedShiftState.selected_subreddit}', 'None')}",
                font_size="0.92em",
                color=ColorScheme.GRAY_11,
                margin_bottom="1rem",
            ),
            # Subreddit grid
            rx.box(
                rx.foreach(
                    FeedShiftState.available_subreddits,
                    lambda subreddit: rx.button(
                        rx.hstack(
                            rx.cond(
                                FeedShiftState.selected_subreddit == subreddit,
                                rx.icon("circle-check", size=12),
                                rx.icon("circle", size=12),
                            ),
                            rx.text(f"r/{subreddit}", font_size="0.88em"),
                            spacing="1",
                            align="center",
                        ),
                        variant=rx.cond(
                            FeedShiftState.selected_subreddit == subreddit,
                            "solid",
                            "soft",
                        ),
                        color_scheme=rx.cond(
                            FeedShiftState.selected_subreddit == subreddit,
                            ColorScheme.PRIMARY,
                            ColorScheme.SECONDARY,
                        ),
                        size="2",
                        margin="0.15rem",
                        cursor="pointer",
                        on_click=FeedShiftState.set_subreddit(subreddit),
                        _hover={"cursor": "pointer"},
                        flex="1 1 auto",
                        min_width="0",
                        width="100%",
                    ),
                ),
                display="grid",
                grid_template_columns="repeat(auto-fit, minmax(140px, 1fr))",
                gap="0.5rem",
                width="100%",
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        margin_bottom="1.25rem",
        padding="1.25rem",
        box_shadow="0 4px 12px rgba(0,0,0,0.05)",
    )


def interest_selector() -> rx.Component:
    """Enhanced interest selector component with toggle functionality."""
    logger.debug("Creating interest selector component")

    return rx.card(
        rx.vstack(
            # Header section
            rx.hstack(
                rx.icon("heart", size=20, color=ColorScheme.PURPLE_9),
                rx.text(
                    "Your Interests",
                    font_size="1.15em",
                    font_weight="600",
                    color=ColorScheme.GRAY_12,
                ),
                align="center",
                spacing="2",
                margin_bottom="0.5rem",
            ),
            # Selection counter
            rx.text(
                f"Selected: {FeedShiftState.selected_interests.length()}",
                font_size="0.92em",
                color=ColorScheme.GRAY_11,
                margin_bottom="1rem",
            ),
            # Interest grid
            rx.box(
                rx.foreach(
                    FeedShiftState.available_interests,
                    lambda interest: rx.button(
                        rx.hstack(
                            rx.cond(
                                FeedShiftState.selected_interests.contains(interest),
                                rx.icon("check", size=12),
                                rx.icon("plus", size=12),
                            ),
                            rx.text(interest, font_size="0.88em"),
                            spacing="1",
                            align="center",
                        ),
                        variant=rx.cond(
                            FeedShiftState.selected_interests.contains(interest),
                            "solid",
                            "soft",
                        ),
                        color_scheme=rx.cond(
                            FeedShiftState.selected_interests.contains(interest),
                            ColorScheme.PRIMARY,
                            ColorScheme.SECONDARY,
                        ),
                        size="2",
                        margin="0.15rem",
                        cursor="pointer",
                        on_click=FeedShiftState.toggle_interest(interest),
                        _hover={"cursor": "pointer"},
                        flex="1 1 auto",
                        min_width="0",
                    ),
                ),
                display="grid",
                grid_template_columns="repeat(auto-fit, minmax(120px, 1fr))",
                gap="0.5rem",
                width="100%",
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        margin_bottom="1.25rem",
        padding="1.25rem",
        box_shadow="0 4px 12px rgba(0,0,0,0.05)",
    )


def toxicity_control() -> rx.Component:
    """Enhanced toxicity control slider with visual feedback."""
    logger.debug("Creating toxicity control component")

    return rx.card(
        rx.vstack(
            # Header section
            rx.hstack(
                rx.icon("shield-check", size=20, color=ColorScheme.PURPLE_9),
                rx.text(
                    "Toxicity Filter",
                    font_size="1.15em",
                    font_weight="600",
                    color=ColorScheme.GRAY_12,
                ),
                align="center",
                spacing="2",
                margin_bottom="0.5rem",
            ),
            # Current value display
            rx.text(
                f"Strictness: {FeedShiftState.toxicity_strictness:.2f}",
                font_size="0.92em",
                color=ColorScheme.GRAY_11,
                margin_bottom="1rem",
            ),
            # Slider control
            rx.slider(
                default_value=[FeedShiftState.toxicity_strictness],
                min_=DefaultValues.TOXICITY_MIN,
                max=DefaultValues.TOXICITY_MAX,
                step=DefaultValues.TOXICITY_STEP,
                on_change=FeedShiftState.set_toxicity_strictness,
                color_scheme=ColorScheme.PRIMARY,
                size="2",
                cursor="pointer",
            ),
            # Descriptive labels
            rx.hstack(
                rx.text("Lenient", font_size="0.85em", color=ColorScheme.GRAY_10),
                rx.spacer(),
                rx.text("Strict", font_size="0.85em", color=ColorScheme.GRAY_10),
                width="100%",
                margin_top="0.5rem",
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        margin_bottom="1.25rem",
        padding="1.25rem",
        box_shadow="0 4px 12px rgba(0,0,0,0.05)",
    )


def diversity_control() -> rx.Component:
    """Enhanced diversity control slider with semantic labels."""
    logger.debug("Creating diversity control component")

    return rx.card(
        rx.vstack(
            # Header section
            rx.hstack(
                rx.icon("shuffle", size=20, color=ColorScheme.PURPLE_9),
                rx.text(
                    "Diversity Control",
                    font_size="1.15em",
                    font_weight="600",
                    color=ColorScheme.GRAY_12,
                ),
                align="center",
                spacing="2",
                margin_bottom="0.5rem",
            ),
            # Current value and label display
            rx.text(
                f"{FeedShiftState.diversity_label}: {FeedShiftState.diversity_strength:.2f}",
                font_size="0.92em",
                color=ColorScheme.GRAY_11,
                margin_bottom="1rem",
            ),
            # Slider control
            rx.slider(
                default_value=[FeedShiftState.diversity_strength],
                min_=DefaultValues.DIVERSITY_MIN,
                max=DefaultValues.DIVERSITY_MAX,
                step=DefaultValues.DIVERSITY_STEP,
                on_change=FeedShiftState.set_diversity_strength,
                color_scheme=ColorScheme.PRIMARY,
                size="2",
                cursor="pointer",
            ),
            # Semantic labels
            rx.hstack(
                rx.text(
                    PostSimilarityType.DIFFERENT,
                    font_size="0.85em",
                    color=ColorScheme.GRAY_10,
                ),
                rx.spacer(),
                rx.text(
                    PostSimilarityType.SAME,
                    font_size="0.85em",
                    color=ColorScheme.GRAY_10,
                ),
                width="100%",
                margin_top="0.5rem",
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        margin_bottom="1.25rem",
        padding="1.25rem",
        box_shadow="0 4px 12px rgba(0,0,0,0.05)",
    )


def action_buttons() -> rx.Component:
    """Action buttons for generating recommendations and resetting state."""
    logger.debug("Creating action buttons component")

    return rx.card(
        rx.vstack(
            # Generate/Regenerate button with dynamic content
            rx.button(
                rx.cond(
                    FeedShiftState.is_processing,
                    rx.hstack(
                        rx.spinner(size="3"),
                        rx.text("Generating..."),
                        spacing="2",
                        align="center",
                    ),
                    rx.cond(
                        FeedShiftState.has_generated_once,
                        rx.hstack(
                            rx.icon("refresh-cw", size=16),
                            rx.text("Regenerate"),
                            spacing="2",
                            align="center",
                        ),
                        rx.hstack(
                            rx.icon("zap", size=16),
                            rx.text("Get Recommendations"),
                            spacing="2",
                            align="center",
                        ),
                    ),
                ),
                on_click=FeedShiftState.generate_recommendations,
                color_scheme=ColorScheme.PRIMARY,
                size="3",
                width="100%",
                disabled=~FeedShiftState.can_generate_recommendations | FeedShiftState.is_processing,
                cursor="pointer",
                _hover={"cursor": "pointer"},
                padding="0.75rem 1.5rem",
                margin_bottom="0.75rem",
            ),
            # Auto-update indicator
            rx.cond(
                FeedShiftState.has_generated_once,
                rx.text(
                    "🔄 Auto-updates enabled",
                    font_size="0.85em",
                    color=ColorScheme.PURPLE_11,
                    text_align="center",
                    margin_bottom="0.5rem",
                    font_style="italic",
                ),
                rx.fragment(),
            ),
            # Reset button
            rx.button(
                rx.hstack(
                    rx.icon("refresh-ccw", size=14),
                    rx.text("Reset All Selections"),
                    spacing="2",
                    align="center",
                ),
                on_click=FeedShiftState.reset_all_selections,
                variant="soft",
                color_scheme=ColorScheme.SECONDARY,
                size="2",
                width="100%",
                cursor="pointer",
                _hover={"cursor": "pointer"},
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        padding="1.25rem",
        box_shadow="0 4px 12px rgba(0,0,0,0.05)",
        background=f"rgba({ColorScheme.PURPLE_1}, 0.5)",
        border=f"1px solid {ColorScheme.PURPLE_7}",
    )


def sidebar() -> rx.Component:
    """Main sidebar with all control components and scrolling capability."""
    logger.debug("Creating main sidebar component")

    return rx.box(
        rx.scroll_area(
            rx.vstack(
                # Sidebar header
                rx.hstack(
                    rx.icon("settings", size=24, color=ColorScheme.PURPLE_9),
                    rx.text(
                        "Customize Feed",
                        font_size="1.3em",
                        font_weight="700",
                        color=ColorScheme.GRAY_12,
                    ),
                    align="center",
                    spacing="2",
                    margin_bottom="1.5rem",
                ),
                # Control components
                subreddit_selector(),
                interest_selector(),
                toxicity_control(),
                diversity_control(),
                action_buttons(),
                spacing="0",
                width="100%",
                padding="0 0.5rem 0 0",
            ),
            height="calc(100vh - 4rem)",
            scrollbars="vertical",
            width="100%",
        ),
        width=UIConstants.SIDEBAR_WIDTH,
        min_width=UIConstants.SIDEBAR_MIN_WIDTH,
        height="calc(100vh - 4rem)",
        position="fixed",
        top="2rem",
        left="2rem",
        padding="1.5rem",
        background=f"rgba({ColorScheme.PURPLE_1}, 0.3)",
        border_radius="0 16px 16px 0",
        border=f"1px solid {ColorScheme.PURPLE_6}",
        box_shadow="0 6px 20px rgba(0,0,0,0.08)",
        z_index="100",
    )


# === Content Display Components ===


def processing_overlay() -> rx.Component:
    """Processing overlay with status information."""
    logger.debug("Creating processing overlay component")

    return rx.cond(
        FeedShiftState.is_processing,
        rx.center(
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.spinner(size="3", color=ColorScheme.PURPLE_9),
                        rx.text(
                            rx.cond(
                                FeedShiftState.has_generated_once,
                                "Updating recommendations...",
                                "Generating recommendations...",
                            ),
                            font_weight="600",
                            color=ColorScheme.PURPLE_11,
                            font_size="1.1em",
                        ),
                        spacing="3",
                        align="center",
                    ),
                    rx.text(
                        FeedShiftState.processing_info,
                        font_size="0.95em",
                        color=ColorScheme.GRAY_11,
                        text_align="center",
                        margin_top="0.5rem",
                    ),
                    spacing="3",
                    align="center",
                ),
                padding="1.75rem 2.5rem",
                background=ColorScheme.PURPLE_2,
                border=f"1px solid {ColorScheme.PURPLE_7}",
                border_radius="14px",
            ),
            width="100%",
            margin="1.5rem 0",
        ),
        rx.fragment(),
    )


def post_card(post: Dict[str, Any], index: int) -> rx.Component:
    """Enhanced post card component with comprehensive content display."""
    logger.debug(f"Creating post card for index {index}")

    # Extract content with fallback logic
    content = rx.cond(
        post.get(ContentFields.CONTENT_FIELDS[0]),
        post.get(ContentFields.CONTENT_FIELDS[0]),
        rx.cond(
            post.get(ContentFields.CONTENT_FIELDS[1]),
            post.get(ContentFields.CONTENT_FIELDS[1]),
            ContentFields.FALLBACK_CONTENT,
        ),
    )

    # Extract author with fallback logic
    author = rx.cond(
        post.get(ContentFields.AUTHOR_FIELDS[0]),
        post.get(ContentFields.AUTHOR_FIELDS[0]),
        rx.cond(
            post.get(ContentFields.AUTHOR_FIELDS[1]),
            post.get(ContentFields.AUTHOR_FIELDS[1]),
            rx.cond(
                post.get(ContentFields.AUTHOR_FIELDS[2]),
                post.get(ContentFields.AUTHOR_FIELDS[2]),
                ContentFields.FALLBACK_AUTHOR,
            ),
        ),
    )

    return rx.card(
        rx.vstack(
            # Header with rank and metadata
            rx.hstack(
                rx.badge(
                    index + 1,
                    color_scheme=ColorScheme.PRIMARY,
                    variant="soft",
                    padding="0.25rem 0.75rem",
                    font_size="0.9em",
                ),
                rx.spacer(),
                rx.text(
                    f"🕒 {post.get(RedditDataCols.TIMESTAMP)} • 🌐 r/{FeedShiftState.selected_subreddit}",
                    font_size="0.9em",
                    color=ColorScheme.GRAY_11,
                ),
                width="100%",
                align="center",
                margin_bottom="0.5rem",
            ),
            # Main content with overflow handling
            rx.text(
                content,
                color=ColorScheme.GRAY_12,
                line_height="1.6",
                font_size="0.95em",
                overflow="hidden",
                text_overflow="ellipsis",
                max_height="6em",
                display="-webkit-box",
                webkit_line_clamp="4",
                webkit_box_orient="vertical",
            ),
            # Footer with score and author
            rx.hstack(
                rx.text(
                    rx.cond(
                        post.get(DataCols.RECOMMENDATION_SCORE),
                        f"Score: {post.get(DataCols.RECOMMENDATION_SCORE)}",
                        "Score: N/A",
                    ),
                    font_weight="600",
                    color=ColorScheme.PURPLE_11,
                    font_size="0.9em",
                ),
                rx.spacer(),
                rx.text(
                    f"Author: {author}",
                    color=ColorScheme.GRAY_10,
                    font_size="0.85em",
                ),
                width="100%",
                align="center",
                margin_top="0.75rem",
            ),
            spacing="3",
            align="start",
        ),
        padding="1.75rem",
        margin="0.85rem 0",
        variant="surface",
        border_left=f"4px solid {ColorScheme.PURPLE_7}",
        width="100%",
        min_height="180px",
        _hover={
            "box_shadow": "0 6px 14px rgba(0,0,0,0.08)",
            "transform": UIConstants.CARD_HOVER_TRANSFORM,
            "transition": f"all {UIConstants.HOVER_TRANSITION_DURATION} ease",
        },
    )


def welcome_section() -> rx.Component:
    """Welcome section shown when no recommendations are generated yet."""
    logger.debug("Creating welcome section component")

    return rx.center(
        rx.vstack(
            # Logo and branding
            rx.image(
                src=UIConstants.LOGO_FILENAME,
                alt="Feedshift Logo",
                height=UIConstants.LOGO_HEIGHT,
                object_fit="contain",
                margin_bottom="1.5rem",
            ),
            rx.heading(
                "FeedShift Recommendation Engine",
                size="5",
                color=ColorScheme.PURPLE_11,
                margin_bottom="1rem",
            ),
            rx.text(
                "Select a subreddit, "
                + "customize your preferences, and click 'Get Recommendations' to see personalized posts",
                color=ColorScheme.GRAY_11,
                font_size="1.05em",
                margin_bottom="2rem",
                text_align="center",
                max_width="600px",
            ),
            # Instructions card
            rx.card(
                rx.vstack(
                    rx.text(
                        "How to use:",
                        font_size="1.1em",
                        font_weight="600",
                        color=ColorScheme.GRAY_12,
                        margin_bottom="0.75rem",
                    ),
                    rx.vstack(
                        rx.hstack(
                            rx.icon("circle-check", size=16, color=ColorScheme.PURPLE_9),
                            rx.text("1. Choose a subreddit from the sidebar", color=ColorScheme.GRAY_11),
                            spacing="2",
                            align="center",
                        ),
                        rx.hstack(
                            rx.icon("circle-check", size=16, color=ColorScheme.PURPLE_9),
                            rx.text("2. Select your interests (optional)", color=ColorScheme.GRAY_11),
                            spacing="2",
                            align="center",
                        ),
                        rx.hstack(
                            rx.icon("circle-check", size=16, color=ColorScheme.PURPLE_9),
                            rx.text("3. Adjust toxicity and diversity filters", color=ColorScheme.GRAY_11),
                            spacing="2",
                            align="center",
                        ),
                        rx.hstack(
                            rx.icon("circle-check", size=16, color=ColorScheme.PURPLE_9),
                            rx.text("4. Click 'Get Recommendations' to see results", color=ColorScheme.GRAY_11),
                            spacing="2",
                            align="center",
                        ),
                        rx.hstack(
                            rx.icon("zap", size=16, color=ColorScheme.PURPLE_9),
                            rx.text(
                                "5. After first generation, filters auto-update results!",
                                color=ColorScheme.PURPLE_11,
                                font_weight="600",
                            ),
                            spacing="2",
                            align="center",
                        ),
                        spacing="1",
                        align="start",
                    ),
                    spacing="2",
                    align="start",
                ),
                padding="1.5rem",
                background=ColorScheme.PURPLE_2,
                border=f"1px solid {ColorScheme.PURPLE_6}",
                max_width="500px",
            ),
            align="center",
            spacing="2",
            width="100%",
        ),
        width="100%",
        flex="1",
        padding="2rem",
    )


def results_section() -> rx.Component:
    """Results section shown when recommendations are available."""
    logger.debug("Creating results section component")

    return rx.vstack(
        # Compact logo and branding for results view
        rx.center(
            rx.vstack(
                rx.image(
                    src=UIConstants.LOGO_FILENAME,
                    alt="Feedshift Logo",
                    height=UIConstants.LOGO_HEIGHT_SMALL,
                    object_fit="contain",
                    margin_bottom="0.75rem",
                ),
                rx.text(
                    "FeedShift Recommendations",
                    font_size="1.2em",
                    font_weight="600",
                    color=ColorScheme.PURPLE_11,
                    margin_bottom="1.5rem",
                ),
                spacing="1",
                align="center",
            ),
            margin_bottom="1rem",
        ),
        # Main header section
        rx.hstack(
            rx.icon("trending-up", size=30, color=ColorScheme.PURPLE_9),
            rx.heading(
                f"Top Posts from r/{FeedShiftState.selected_subreddit}",
                size="6",
                color=ColorScheme.GRAY_12,
            ),
            align="center",
            spacing="3",
            justify="center",
        ),
        rx.text(
            "Showing top posts tailored to your preferences",
            color=ColorScheme.GRAY_11,
            font_size="1.05em",
            margin_bottom="1.25rem",
            text_align="center",
        ),
        # Processing overlay
        processing_overlay(),
        # Results content with conditional display
        rx.cond(
            FeedShiftState.ranked_posts.length() > 0,
            rx.cond(
                ~FeedShiftState.is_processing,
                rx.scroll_area(
                    rx.vstack(
                        rx.foreach(FeedShiftState.ranked_posts, post_card),
                        spacing="3",
                        width="100%",
                        align="center",
                    ),
                    height="65vh",
                    scrollbars="vertical",
                    width="100%",
                    min_width="700px",
                ),
                rx.fragment(),
            ),
            rx.center(
                rx.text(
                    "No posts available. Try adjusting your filters or selecting different interests.",
                    color=ColorScheme.GRAY_11,
                    font_size="1.1em",
                    text_align="center",
                ),
                padding="2.5rem",
            ),
        ),
        spacing="2",
        width="100%",
        max_width=UIConstants.CARD_MAX_WIDTH,
        align="center",
    )


def main_content() -> rx.Component:
    """Main content area with conditional display logic."""
    return rx.box(
        rx.center(
            rx.cond(
                FeedShiftState.ranked_posts.length() > 0,
                results_section(),
                welcome_section(),
            ),
            width="100%",
        ),
        width="100%",
        flex="1",
        margin_left="140px",
        padding_right="2rem",
    )
