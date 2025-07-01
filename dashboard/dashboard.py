import io
import asyncio

import pandas as pd
import reflex as rx

from src.engine.run_engine import FeedShiftEngine
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
    uploaded_content: str = ""
    toxicity_strictness: float = DefaultValues.TOXICITY_STRICTNESS
    diversity_strength: float = 0.5  # New diversity parameter
    ranked_data: pd.DataFrame | None = None
    ranked_posts: list[dict] = []
    file_upload: bool = False
    file_selected: bool = False  # New state to track file selection
    is_processing: bool = False
    _current_task_id: int = 0
    _last_processed_value: float = DefaultValues.TOXICITY_STRICTNESS
    _last_processed_diversity: float = 0.5
    _cancelled_tasks: int = 0  # Track cancelled tasks for debugging

    # Interest selection state
    selected_interests: list[str] = []
    available_interests: list[str] = InterestCategories.get_all()

    def __init__(self, **kwargs):
        """Initialize state and reset on page load"""
        super().__init__(**kwargs)
        # Always reset to initial state when component initializes
        self.reset_to_initial_state()

    def reset_to_initial_state(self):
        """Reset all state to initial values"""
        self.data = None
        self.uploaded_content = ""
        self.toxicity_strictness = DefaultValues.TOXICITY_STRICTNESS
        self.diversity_strength = 0.5
        self.ranked_data = None
        self.ranked_posts = []
        self.file_upload = False
        self.file_selected = False  # Reset file selection state
        self.is_processing = False
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
        return f"Processing with {interests_text}, toxicity: {self.toxicity_strictness:.2f}, diversity: {self.diversity_label.lower()} ({self.diversity_strength:.2f})"

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            return

        # Set file_selected to True when files are present
        self.file_selected = True
        self.is_processing = True
        yield

        try:
            file = files[0]
            if file.name.endswith(".csv"):
                content = await file.read()
                self.data = pd.read_csv(io.StringIO(content.decode("utf-8")))
                self.uploaded_content = self.data.head().to_markdown()

                if self.data is not None:
                    self.file_upload = True
                    self.ranked_data = get_recommended_posts(
                        self.data,
                        self.selected_interests,
                        self.toxicity_strictness,
                        self.diversity_strength,
                    )
                    self.ranked_posts = self.ranked_data.to_dict("records")
                    self._last_processed_value = self.toxicity_strictness
                    self._last_processed_diversity = self.diversity_strength
        except Exception as e:
            print(f"Error uploading file: {e}")
        finally:
            self.is_processing = False

    @rx.event
    def toggle_interest(self, interest: str):
        """Toggle interest selection on/off"""
        if interest in self.selected_interests:
            self.selected_interests = [i for i in self.selected_interests if i != interest]
        else:
            self.selected_interests = self.selected_interests + [interest]

        # Auto-update recommendations if file is uploaded
        if self.data is not None and self.file_upload:
            self._current_task_id += 1
            current_task = self._current_task_id
            return FeedShiftState.update_recommendations_delayed(current_task)

    @rx.event
    def set_toxicity_strictness(self, value):
        if isinstance(value, list):
            new_strictness = float(value[0]) if value else DefaultValues.TOXICITY_STRICTNESS
        else:
            new_strictness = float(value)

        self.toxicity_strictness = new_strictness

        # Only process if we have data and file is uploaded
        if self.data is not None and self.file_upload:
            # Increment task ID to cancel all previous pending tasks
            self._current_task_id += 1
            current_task = self._current_task_id

            # Schedule the update with the current task ID
            return FeedShiftState.update_recommendations_delayed(current_task)

    @rx.event
    def set_diversity_strength(self, value):
        """Set diversity strength value"""
        if isinstance(value, list):
            new_diversity = float(value[0]) if value else 0.5
        else:
            new_diversity = float(value)

        self.diversity_strength = new_diversity

        # Only process if we have data and file is uploaded
        if self.data is not None and self.file_upload:
            # Increment task ID to cancel all previous pending tasks
            self._current_task_id += 1
            current_task = self._current_task_id

            # Schedule the update with the current task ID
            return FeedShiftState.update_recommendations_delayed(current_task)

    @rx.event
    async def update_recommendations_delayed(self, task_id: int):
        """Update recommendations with proper debouncing using task ID"""
        # Wait for debounce period
        await asyncio.sleep(UIConstants.DEBOUNCE_DELAY)

        # Check if this task is still the current one (early exit to avoid processing)
        if self._current_task_id != task_id:
            self._cancelled_tasks += 1
            return  # A newer request came in, abort this one

        # Check if we still have data and should process
        if self.data is None or not self.file_upload:
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
                f"Processing task {task_id} with toxicity {self.toxicity_strictness} and diversity {self.diversity_strength} with interests {self.selected_interests}"
            )
            # Perform the actual processing with interests and diversity
            self.ranked_data = get_recommended_posts(
                self.data,
                self.selected_interests,
                self.toxicity_strictness,
                self.diversity_strength,
            )
            self.ranked_posts = self.ranked_data.to_dict("records")
            self._last_processed_value = self.toxicity_strictness
            self._last_processed_diversity = self.diversity_strength
        except Exception as e:
            print(f"Error updating recommendations: {e}")
        finally:
            self.is_processing = False

    @rx.event
    def reset_upload(self):
        """Reset upload state"""
        self.reset_to_initial_state()


def get_recommended_posts(
    data: pd.DataFrame,
    selected_interests: list[str],
    toxicity_strictness: float,
    diversity_strength: float,
) -> pd.DataFrame:
    engine = FeedShiftEngine(data)
    # Pass diversity_strength to the engine - you'll need to modify your engine to accept this parameter
    ranked_data = engine.run(selected_interests, toxicity_strictness, diversity_strength).head(
        UIConstants.TOP_POSTS_LIMIT
    )
    return ranked_data


def upload_zone():
    """Upload zone that changes size based on upload state"""
    return rx.center(
        rx.box(
            rx.upload(
                rx.cond(
                    FeedShiftState.file_upload,
                    # Compact view after upload - FIXED: Smaller size
                    rx.hstack(
                        rx.icon(
                            "file-check",
                            size=20,
                            color=ColorScheme.PURPLE_9,
                        ),
                        rx.text(
                            "File uploaded",
                            font_size="0.9em",
                            font_weight="600",
                            color=ColorScheme.GRAY_12,
                        ),
                        rx.button(
                            rx.hstack(
                                rx.icon("refresh-ccw", size=12),
                                rx.text("New Upload"),
                                spacing="1",
                            ),
                            on_click=FeedShiftState.reset_upload,
                            variant="soft",
                            color_scheme=ColorScheme.SECONDARY,
                            size="1",
                            cursor="pointer",
                            _hover={"cursor": "pointer"},
                        ),
                        align="center",
                        spacing="3",
                        padding="0.75rem 1rem",
                    ),
                    # Full upload view
                    rx.vstack(
                        rx.icon(
                            "cloud-upload",
                            size=32,
                            color=ColorScheme.PURPLE_9,
                        ),
                        rx.text(
                            "Drop CSV file here",
                            font_size="1em",
                            font_weight="600",
                            color=ColorScheme.GRAY_12,
                        ),
                        rx.text(
                            "or click to browse",
                            font_size="0.8em",
                            color=ColorScheme.GRAY_11,
                        ),
                        rx.button(
                            "Choose File",
                            variant="soft",
                            color_scheme=ColorScheme.PRIMARY,
                            size="2",
                            margin_top="0.5rem",
                            cursor="pointer",
                        ),
                        align="center",
                        spacing="2",
                        padding="2rem",
                    ),
                ),
                id="file_upload",
                max_files=UIConstants.MAX_FILES,
                border=f"2px dashed {ColorScheme.PURPLE_8}",
                border_radius="12px",
                background=ColorScheme.PURPLE_2,
                transition="all 0.2s ease",
                cursor="pointer",
                _hover={
                    "border_color": ColorScheme.PURPLE_9,
                    "background": ColorScheme.PURPLE_3,
                    "cursor": "pointer",
                },
                width=rx.cond(
                    FeedShiftState.file_upload,
                    "auto",  # FIXED: Auto width when uploaded
                    "100%",
                ),
                min_height=rx.cond(
                    FeedShiftState.file_upload,
                    "auto",  # FIXED: Auto height when uploaded
                    UIConstants.UPLOAD_ZONE_MIN_HEIGHT,
                ),
            ),
            width=rx.cond(
                FeedShiftState.file_upload,
                "auto",  # FIXED: Auto width when uploaded
                "100%",
            ),
        ),
        width="100%",
    )


def interest_selector():
    """Interest selection component for sidebar - FIXED: Grid layout with larger text"""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("heart", size=20, color=ColorScheme.PURPLE_9),
                rx.text(
                    "Your Interests",
                    font_size="1.1em",
                    font_weight="600",
                    color=ColorScheme.GRAY_12,
                ),
                align="center",
                spacing="2",
            ),
            rx.text(
                f"Selected: {FeedShiftState.selected_interests.length()}",
                font_size="0.9em",
                color=ColorScheme.GRAY_11,
                margin_bottom="0.5rem",
            ),
            # FIXED: Grid layout for interests instead of vertical
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
                            rx.text(interest, font_size="0.85em"),
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
                        margin="0.1rem",
                        cursor="pointer",
                        on_click=FeedShiftState.toggle_interest(interest),
                        _hover={"cursor": "pointer"},
                        flex="1 1 auto",
                        min_width="0",
                    ),
                ),
                display="grid",
                grid_template_columns="repeat(auto-fit, minmax(110px, 1fr))",
                gap="0.4rem",
                width="100%",
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        margin_bottom="1rem",
    )


def toxicity_control():
    """Toxicity control for sidebar"""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("shield-check", size=20, color=ColorScheme.PURPLE_9),
                rx.text(
                    "Toxicity Filter",
                    font_size="1.1em",
                    font_weight="600",
                    color=ColorScheme.GRAY_12,
                ),
                align="center",
                spacing="2",
            ),
            rx.text(
                f"Strictness: {FeedShiftState.toxicity_strictness:.2f}",
                font_size="0.9em",
                color=ColorScheme.GRAY_11,
                margin_bottom="0.5rem",
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
                rx.text("Lenient", font_size="0.8em", color=ColorScheme.GRAY_10),
                rx.spacer(),
                rx.text("Strict", font_size="0.8em", color=ColorScheme.GRAY_10),
                width="100%",
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        margin_bottom="1rem",
    )


def diversity_control():
    """Diversity control for sidebar"""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("shuffle", size=20, color=ColorScheme.PURPLE_9),
                rx.text(
                    "Diversity Control",
                    font_size="1.1em",
                    font_weight="600",
                    color=ColorScheme.GRAY_12,
                ),
                align="center",
                spacing="2",
            ),
            rx.text(
                f"{FeedShiftState.diversity_label}: {FeedShiftState.diversity_strength:.2f}",
                font_size="0.9em",
                color=ColorScheme.GRAY_11,
                margin_bottom="0.5rem",
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
                    font_size="0.8em",
                    color=ColorScheme.GRAY_10,
                ),
                rx.spacer(),
                rx.text(
                    PostSimilarityType.SAME,
                    font_size="0.8em",
                    color=ColorScheme.GRAY_10,
                ),
                width="100%",
            ),
            spacing="2",
            width="100%",
        ),
        width="100%",
        margin_bottom="1rem",
    )


def sidebar():
    """Left sidebar with all controls - always visible"""
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.icon("settings", size=22, color=ColorScheme.PURPLE_9),
                rx.text(
                    "Customize Feed",
                    font_size="1.2em",
                    font_weight="700",
                    color=ColorScheme.GRAY_12,
                ),
                align="center",
                spacing="2",
                margin_bottom="1rem",
            ),
            interest_selector(),
            toxicity_control(),
            diversity_control(),
            spacing="0",
            width="100%",
        ),
        width="320px",
        min_width="320px",
        height="fit-content",
        position="sticky",
        top="2rem",
        padding="1.2rem",
        background=f"rgba({ColorScheme.PURPLE_1}, 0.3)",
        border_radius="12px",
        border=f"1px solid {ColorScheme.PURPLE_6}",
    )


def process_button():
    return rx.center(
        rx.cond(
            FeedShiftState.file_upload,
            rx.fragment(),
            rx.cond(
                rx.selected_files("file_upload"),
                rx.button(
                    rx.cond(
                        FeedShiftState.is_processing,
                        rx.hstack(
                            rx.spinner(size="3"),
                            rx.text("Processing..."),
                            spacing="2",
                            align="center",
                        ),
                        rx.hstack(
                            rx.icon("play", size=16),
                            rx.text("Generate Recommendations"),
                            spacing="2",
                            align="center",
                        ),
                    ),
                    on_click=FeedShiftState.handle_upload(rx.upload_files("file_upload")),
                    color_scheme=ColorScheme.PRIMARY,
                    size="3",
                    width="300px",
                    disabled=FeedShiftState.is_processing,
                    cursor="pointer",
                    _hover={"cursor": "pointer"},
                ),
                rx.fragment(),
            ),
        ),
        width="100%",
        margin_top="1rem",
    )


def processing_overlay():
    """Show processing spinner when updating recommendations with detailed info"""
    return rx.cond(
        FeedShiftState.is_processing & FeedShiftState.file_upload,
        rx.center(
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.spinner(size="3", color=ColorScheme.PURPLE_9),
                        rx.text(
                            "Updating recommendations...",
                            font_weight="600",
                            color=ColorScheme.PURPLE_11,
                        ),
                        spacing="3",
                        align="center",
                    ),
                    rx.text(
                        FeedShiftState.processing_info,
                        font_size="0.9em",
                        color=ColorScheme.GRAY_11,
                        text_align="center",
                    ),
                    spacing="3",
                    align="center",
                ),
                padding="1.5rem 2rem",
                background=ColorScheme.PURPLE_2,
                border=f"1px solid {ColorScheme.PURPLE_7}",
            ),
            width="100%",
            margin="1rem 0",
        ),
        rx.fragment(),
    )


def post_card(post, index):
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.badge(
                    index + 1,
                    color_scheme=ColorScheme.PRIMARY,
                    variant="soft",
                ),
                rx.spacer(),
                rx.text(
                    f"🕒 {post.get('timestamp')} • 🌐 {post.get('platform')}",
                    font_size="14px",
                    color=ColorScheme.GRAY_11,
                ),
                width="100%",
                align="center",
            ),
            rx.text(
                rx.cond(
                    post.get(ContentFields.CONTENT_FIELDS[0]),
                    post.get(ContentFields.CONTENT_FIELDS[0]),
                    rx.cond(
                        post.get(ContentFields.CONTENT_FIELDS[1]),
                        post.get(ContentFields.CONTENT_FIELDS[1]),
                        rx.cond(
                            post.get(ContentFields.CONTENT_FIELDS[2]),
                            post.get(ContentFields.CONTENT_FIELDS[2]),
                            ContentFields.FALLBACK_CONTENT,
                        ),
                    ),
                ),
                color=ColorScheme.GRAY_12,
                line_height="1.5",
                font_size="1em",
            ),
            rx.hstack(
                rx.text(
                    rx.cond(
                        post.get("scores"),
                        f"Score: {post.get('scores')}",
                        "Score: N/A",
                    ),
                    font_weight="600",
                    color=ColorScheme.PURPLE_11,
                    font_size="0.9em",
                ),
                rx.spacer(),
                rx.text(
                    rx.cond(
                        post.get(ContentFields.AUTHOR_FIELDS[0]),
                        f"Author: {post.get(ContentFields.AUTHOR_FIELDS[0])}",
                        rx.cond(
                            post.get(ContentFields.AUTHOR_FIELDS[1]),
                            f"Author: {post.get(ContentFields.AUTHOR_FIELDS[1])}",
                            rx.cond(
                                post.get(ContentFields.AUTHOR_FIELDS[2]),
                                f"Author: {post.get(ContentFields.AUTHOR_FIELDS[2])}",
                                ContentFields.FALLBACK_AUTHOR,
                            ),
                        ),
                    ),
                    color=ColorScheme.GRAY_10,
                    font_size="0.8em",
                ),
                width="100%",
                align="center",
            ),
            spacing="3",
            align="start",
        ),
        padding="1.5rem",
        margin="0.75rem 0",
        variant="surface",
        border_left=f"4px solid {ColorScheme.PURPLE_7}",
        width="100%",
    )


def upload_section():
    """Upload section with full width when no file uploaded"""
    return rx.cond(
        ~FeedShiftState.file_upload,
        rx.center(
            rx.vstack(
                upload_zone(),
                rx.center(
                    rx.text(
                        rx.selected_files("file_upload"),
                        color=ColorScheme.PURPLE_11,
                        font_weight="500",
                    ),
                    width="100%",
                ),
                process_button(),
                spacing="4",
                width="100%",
                max_width="600px",
            ),
            width="100%",
            flex="1",
        ),
        rx.fragment(),
    )


def main_content():
    """Main content area with recommended posts - FIXED: Centered layout with spacing"""
    return rx.cond(
        FeedShiftState.file_upload,
        rx.center(  # FIXED: Center the entire content
            rx.box(
                rx.vstack(
                    upload_zone(),
                    rx.spacer(height="2rem"),  # FIXED: Add space between upload and posts
                    rx.hstack(
                        rx.icon("trending-up", size=28, color=ColorScheme.PURPLE_9),
                        rx.heading(
                            "Top 10 Recommended Posts",
                            size="7",
                            color=ColorScheme.GRAY_12,
                        ),
                        align="center",
                        spacing="3",
                        justify="center",  # FIXED: Center the heading
                    ),
                    rx.text(
                        f"Showing top {UIConstants.TOP_POSTS_LIMIT} posts tailored to your preferences",
                        color=ColorScheme.GRAY_11,
                        font_size="1em",
                        margin_bottom="1rem",
                        text_align="center",  # FIXED: Center the text
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
                                    align="center",  # FIXED: Center the posts
                                ),
                                height="70vh",
                                scrollbars="vertical",
                                width="100%",
                            ),
                            rx.fragment(),
                        ),
                        rx.center(
                            rx.text(
                                "No posts available",
                                color=ColorScheme.GRAY_11,
                                font_size="1em",
                            ),
                            padding="2rem",
                        ),
                    ),
                    spacing="2",
                    width="100%",
                    max_width="700px",  # FIXED: Slightly narrower for better alignment
                    align="center",  # FIXED: Center all items
                ),
                width="100%",
                display="flex",
                justify_content="center",
                margin_left="-40px",  # FIXED: Move content slightly left to center with logo
            ),
            width="100%",
            flex="1",
        ),
        upload_section(),
    )


def header():
    return rx.center(
        rx.vstack(
            rx.image(
                src=UIConstants.LOGO_FILENAME,
                alt="Feedshift Logo",
                height=UIConstants.LOGO_HEIGHT,
                object_fit="contain",
            ),
            spacing="4",
            align="center",
            margin_bottom="2rem",
        ),
        width="100%",
    )


def index():
    return rx.box(
        rx.color_mode.button(position="fixed", top="1rem", right="1rem", z_index="999"),
        rx.vstack(
            header(),
            # Main layout with sidebar always visible - full width
            rx.hstack(
                sidebar(),
                main_content(),
                spacing="2",
                align="start",
                width="100%",
                padding="0 2rem",
            ),
            spacing="2",
            width="100%",
            max_width="100%",
        ),
        background=f"radial-gradient(ellipse at top, {ColorScheme.PURPLE_3}, transparent), radial-gradient(ellipse at bottom, {ColorScheme.PURPLE_2}, transparent)",
        min_height="100vh",
        width="100%",
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
