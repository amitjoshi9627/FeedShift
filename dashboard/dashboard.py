import asyncio
from pathlib import Path

import pandas as pd
import reflex as rx

from src.data.constants import RedditDataCols, DataCols, SubRedditList
from src.engine.constants import Platform
from src.engine.engine_factory import get_engine
from dashboard.constants import (
    InterestCategories,
    UIConstants,
    DefaultValues,
    ContentFields,
    ColorScheme,
    PostSimilarityType,
)


class FeedShiftState(rx.State):
    data: pd.DataFrame | None = None
    saved_data_path: str | Path = ""
    selected_subreddit: str = ""
    toxicity_strictness: float = DefaultValues.TOXICITY_STRICTNESS
    diversity_strength: float = 0.5
    ranked_data: pd.DataFrame | None = None
    ranked_posts: list[dict] = []
    subreddit_selected: bool = False
    is_processing: bool = False
    has_generated_once: bool = False  # Track if recommendations have been generated at least once
    _current_task_id: int = 0
    _last_processed_value: float = DefaultValues.TOXICITY_STRICTNESS
    _last_processed_diversity: float = 0.5
    _cancelled_tasks: int = 0

    # Interest selection state
    selected_interests: list[str] = []
    available_interests: list[str] = InterestCategories.get_all()
    available_subreddits: list[str] = SubRedditList.get_all()

    def __init__(self, **kwargs):
        """Initialize state and reset on page load"""
        super().__init__(**kwargs)
        self.reset_to_initial_state()

    def reset_to_initial_state(self):
        """Reset all state to initial values"""
        self.data = None
        self.saved_data_path = ""
        self.selected_subreddit = ""
        self.toxicity_strictness = DefaultValues.TOXICITY_STRICTNESS
        self.diversity_strength = 0.5
        self.ranked_data = None
        self.ranked_posts = []
        self.subreddit_selected = False
        self.is_processing = False
        self.has_generated_once = False
        self._current_task_id = 0
        self._last_processed_value = DefaultValues.TOXICITY_STRICTNESS
        self._last_processed_diversity = 0.5
        self._cancelled_tasks = 0
        self.selected_interests = []

    @rx.var
    def diversity_label(self) -> str:
        """Get diversity label based on current diversity strength value"""
        if self.diversity_strength < 0.4:
            return PostSimilarityType.DIFFERENT
        elif self.diversity_strength < 0.6:
            return PostSimilarityType.DIVERSE
        elif self.diversity_strength < 0.8:
            return PostSimilarityType.NEAR
        elif self.diversity_strength < 1.0:
            return PostSimilarityType.SIMILAR
        else:
            return PostSimilarityType.SAME

    @rx.var
    def processing_info(self) -> str:
        """Get current processing information"""
        interests_text = f"{len(self.selected_interests)} interests" if self.selected_interests else "No interests"
        subreddit_text = f"r/{self.selected_subreddit}" if self.selected_subreddit else "No subreddit"
        return f"Processing {subreddit_text} with {interests_text}, toxicity: {self.toxicity_strictness:.2f}, diversity: {self.diversity_label.lower()} ({self.diversity_strength:.2f})"

    @rx.var
    def can_generate_recommendations(self) -> bool:
        """Check if we can generate recommendations"""
        return bool(self.selected_subreddit)

    @rx.event
    def set_subreddit(self, subreddit: str):
        """Set selected subreddit"""
        self.selected_subreddit = subreddit
        self.subreddit_selected = True
        # Clear previous results when changing subreddit
        self.ranked_data = None
        self.ranked_posts = []

        # Auto-generate if we've generated recommendations before
        if self.has_generated_once:
            self._current_task_id += 1
            current_task = self._current_task_id
            return FeedShiftState.auto_generate_recommendations(current_task)

    @rx.event
    def toggle_interest(self, interest: str):
        """Toggle interest selection on/off"""
        if interest in self.selected_interests:
            self.selected_interests = [i for i in self.selected_interests if i != interest]
        else:
            self.selected_interests = self.selected_interests + [interest]

        # Auto-generate if we've generated recommendations before
        if self.has_generated_once and self.selected_subreddit:
            self._current_task_id += 1
            current_task = self._current_task_id
            return FeedShiftState.auto_generate_recommendations_delayed(current_task)

    @rx.event
    def set_toxicity_strictness(self, value):
        if isinstance(value, list):
            new_strictness = float(value[0]) if value else DefaultValues.TOXICITY_STRICTNESS
        else:
            new_strictness = float(value)
        self.toxicity_strictness = new_strictness

        # Auto-generate if we've generated recommendations before
        if self.has_generated_once and self.selected_subreddit:
            self._current_task_id += 1
            current_task = self._current_task_id
            return FeedShiftState.auto_generate_recommendations_delayed(current_task)

    @rx.event
    def set_diversity_strength(self, value):
        """Set diversity strength value"""
        if isinstance(value, list):
            new_diversity = float(value[0]) if value else 0.5
        else:
            new_diversity = float(value)
        self.diversity_strength = new_diversity

        # Auto-generate if we've generated recommendations before
        if self.has_generated_once and self.selected_subreddit:
            self._current_task_id += 1
            current_task = self._current_task_id
            return FeedShiftState.auto_generate_recommendations_delayed(current_task)

    @rx.event
    async def generate_recommendations(self):
        """Generate recommendations with current settings (manual trigger)"""
        if not self.selected_subreddit:
            return

        self.is_processing = True
        yield

        try:
            print(f"Generating recommendations for r/{self.selected_subreddit}")

            self.ranked_data = get_recommended_posts(
                self.selected_subreddit,
                self.selected_interests,
                self.toxicity_strictness,
                self.diversity_strength,
            )
            self.ranked_posts = self.ranked_data.to_dict("records")
            self._last_processed_value = self.toxicity_strictness
            self._last_processed_diversity = self.diversity_strength
            self.has_generated_once = True  # Mark that we've generated at least once
        except Exception as e:
            print(f"Error generating recommendations: {e}")
        finally:
            self.is_processing = False

    @rx.event
    async def auto_generate_recommendations(self, task_id: int):
        """Auto-generate recommendations immediately (for subreddit changes)"""
        # Check if this task is still current
        if self._current_task_id != task_id:
            self._cancelled_tasks += 1
            return

        if not self.selected_subreddit:
            return

        self.is_processing = True
        yield

        try:
            # Final check before expensive operation
            if self._current_task_id != task_id:
                self._cancelled_tasks += 1
                return

            print(f"Auto-generating recommendations for r/{self.selected_subreddit}")

            self.ranked_data = get_recommended_posts(
                self.selected_subreddit,
                self.selected_interests,
                self.toxicity_strictness,
                self.diversity_strength,
            )
            self.ranked_posts = self.ranked_data.to_dict("records")
            self._last_processed_value = self.toxicity_strictness
            self._last_processed_diversity = self.diversity_strength
        except Exception as e:
            print(f"Error auto-generating recommendations: {e}")
        finally:
            self.is_processing = False

    @rx.event
    async def auto_generate_recommendations_delayed(self, task_id: int):
        """Auto-generate recommendations with debouncing (for filter changes)"""
        # Wait for debounce period
        await asyncio.sleep(UIConstants.DEBOUNCE_DELAY)

        # Check if this task is still the current one
        if self._current_task_id != task_id:
            self._cancelled_tasks += 1
            return

        # Check if we still have subreddit selected
        if not self.selected_subreddit:
            return

        # Double-check the task ID right before processing
        if self._current_task_id != task_id:
            self._cancelled_tasks += 1
            return

        self.is_processing = True
        yield

        try:
            # Final check before expensive operation
            if self._current_task_id != task_id:
                self._cancelled_tasks += 1
                print(f"Task {task_id} cancelled during processing")
                return

            print(
                f"Auto-updating recommendations (task {task_id}) with toxicity {self.toxicity_strictness} and diversity {self.diversity_strength} for r/{self.selected_subreddit}"
            )

            self.ranked_data = get_recommended_posts(
                self.selected_subreddit,
                self.selected_interests,
                self.toxicity_strictness,
                self.diversity_strength,
            )
            self.ranked_posts = self.ranked_data.to_dict("records")
            self._last_processed_value = self.toxicity_strictness
            self._last_processed_diversity = self.diversity_strength
        except Exception as e:
            print(f"Error auto-updating recommendations: {e}")
        finally:
            self.is_processing = False

    @rx.event
    def reset_all_selections(self):
        """Reset all selections and results"""
        self.reset_to_initial_state()


def get_recommended_posts(
    subreddit: str,
    selected_interests: list[str],
    toxicity_strictness: float,
    diversity_strength: float,
) -> pd.DataFrame:
    # Updated to work with subreddit instead of file path
    engine = get_engine(Platform.REDDIT, subreddit)
    ranked_data = engine.run(selected_interests, toxicity_strictness, diversity_strength).head(
        UIConstants.TOP_POSTS_LIMIT
    )
    return ranked_data


def subreddit_selector():
    """Enhanced subreddit selector with aesthetic design"""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("reddit", size=20, color=ColorScheme.PURPLE_9),
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
            rx.text(
                f"Selected: {rx.cond(FeedShiftState.selected_subreddit, f'r/{FeedShiftState.selected_subreddit}', 'None')}",
                font_size="0.92em",
                color=ColorScheme.GRAY_11,
                margin_bottom="1rem",
            ),
            rx.box(
                rx.foreach(
                    FeedShiftState.available_subreddits,
                    lambda subreddit: rx.button(
                        rx.hstack(
                            rx.cond(
                                FeedShiftState.selected_subreddit == subreddit,
                                rx.icon("check-circle", size=12),
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


def interest_selector():
    """Enhanced interest selector with better spacing"""
    return rx.card(
        rx.vstack(
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
            rx.text(
                f"Selected: {FeedShiftState.selected_interests.length()}",
                font_size="0.92em",
                color=ColorScheme.GRAY_11,
                margin_bottom="1rem",
            ),
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


def toxicity_control():
    """Enhanced toxicity control with better spacing"""
    return rx.card(
        rx.vstack(
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
            rx.text(
                f"Strictness: {FeedShiftState.toxicity_strictness:.2f}",
                font_size="0.92em",
                color=ColorScheme.GRAY_11,
                margin_bottom="1rem",
            ),
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


def diversity_control():
    """Enhanced diversity control with better spacing"""
    return rx.card(
        rx.vstack(
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
            rx.text(
                f"{FeedShiftState.diversity_label}: {FeedShiftState.diversity_strength:.2f}",
                font_size="0.92em",
                color=ColorScheme.GRAY_11,
                margin_bottom="1rem",
            ),
            rx.slider(
                default_value=[FeedShiftState.diversity_strength],
                min_=0.0,
                max=1.0,
                step=0.01,
                on_change=FeedShiftState.set_diversity_strength,
                color_scheme=ColorScheme.PRIMARY,
                size="2",
                cursor="pointer",
            ),
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


def action_buttons():
    """Action buttons for generating recommendations and resetting"""
    return rx.card(
        rx.vstack(
            # Generate Recommendations Button - show different text based on state
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
            # Reset All Button
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


def sidebar():
    """Sidebar fixed to the leftmost side with scrolling capability"""
    return rx.box(
        rx.scroll_area(
            rx.vstack(
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
                subreddit_selector(),
                interest_selector(),
                toxicity_control(),
                diversity_control(),
                action_buttons(),  # Added action buttons at the bottom
                spacing="0",
                width="100%",
                padding="0 0.5rem 0 0",  # Right padding for scrollbar
            ),
            height="calc(100vh - 4rem)",  # Full viewport height minus top padding
            scrollbars="vertical",
            width="100%",
        ),
        width="400px",  # Slightly wider to accommodate scrollbar
        min_width="400px",
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


def processing_overlay():
    """Enhanced processing overlay"""
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


def post_card(post, index):
    """Enhanced post card with better text handling"""
    content = rx.cond(
        post.get(ContentFields.CONTENT_FIELDS[0]),
        post.get(ContentFields.CONTENT_FIELDS[0]),
        rx.cond(
            post.get(ContentFields.CONTENT_FIELDS[1]),
            post.get(ContentFields.CONTENT_FIELDS[1]),
            ContentFields.FALLBACK_CONTENT,
        ),
    )

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
            "transform": "translateY(-2px)",
            "transition": "all 0.2s ease",
        },
    )


def welcome_section():
    """Welcome section shown when no recommendations are generated yet"""
    return rx.center(
        rx.vstack(
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
                "Select a subreddit, customize your preferences, and click 'Get Recommendations' to see personalized posts",
                color=ColorScheme.GRAY_11,
                font_size="1.05em",
                margin_bottom="2rem",
                text_align="center",
                max_width="600px",
            ),
            # Instructions
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
                            rx.icon("check-circle", size=16, color=ColorScheme.PURPLE_9),
                            rx.text("1. Choose a subreddit from the sidebar", color=ColorScheme.GRAY_11),
                            spacing="2",
                            align="center",
                        ),
                        rx.hstack(
                            rx.icon("check-circle", size=16, color=ColorScheme.PURPLE_9),
                            rx.text("2. Select your interests (optional)", color=ColorScheme.GRAY_11),
                            spacing="2",
                            align="center",
                        ),
                        rx.hstack(
                            rx.icon("check-circle", size=16, color=ColorScheme.PURPLE_9),
                            rx.text("3. Adjust toxicity and diversity filters", color=ColorScheme.GRAY_11),
                            spacing="2",
                            align="center",
                        ),
                        rx.hstack(
                            rx.icon("check-circle", size=16, color=ColorScheme.PURPLE_9),
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


def results_section():
    """Results section shown when recommendations are available - now with logo"""
    return rx.vstack(
        # Logo and branding section at the top
        rx.center(
            rx.vstack(
                rx.image(
                    src=UIConstants.LOGO_FILENAME,
                    alt="Feedshift Logo",
                    height="200px",  # Smaller than welcome screen
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
        processing_overlay(),
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
                    height="65vh",  # Slightly reduced to accommodate logo
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
        max_width="800px",
        align="center",
    )


def main_content():
    """Main content area"""
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
        # Add left margin to account for fixed sidebar
        margin_left="140px",
        padding_right="2rem",
    )


def index():
    return rx.box(
        rx.color_mode.button(
            position="fixed",
            top="1.25rem",
            right="1.25rem",
            z_index="999",
            size="3",
        ),
        sidebar(),  # Fixed left sidebar with scrolling
        main_content(),
        background=f"radial-gradient(ellipse at top, {ColorScheme.PURPLE_3}, transparent), radial-gradient(ellipse at bottom, {ColorScheme.PURPLE_2}, transparent)",
        min_height="100vh",
        width="100%",
        padding_top="2rem",
    )


app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        radius="large",
        accent_color=ColorScheme.PRIMARY,
    )
)
app.add_page(index)
