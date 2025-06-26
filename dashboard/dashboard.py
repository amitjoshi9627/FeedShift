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
)


class FeedShiftState(rx.State):
    data: pd.DataFrame | None = None
    uploaded_content: str = ""
    toxicity_strictness: float = DefaultValues.TOXICITY_STRICTNESS
    ranked_data: pd.DataFrame | None = None
    ranked_posts: list[dict] = []
    file_upload: bool = False
    file_selected: bool = False  # New state to track file selection
    is_processing: bool = False
    _current_task_id: int = 0
    _last_processed_value: float = DefaultValues.TOXICITY_STRICTNESS
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
        self.ranked_data = None
        self.ranked_posts = []
        self.file_upload = False
        self.file_selected = False  # Reset file selection state
        self.is_processing = False
        self._current_task_id = 0
        self._last_processed_value = DefaultValues.TOXICITY_STRICTNESS
        self._cancelled_tasks = 0
        self.selected_interests = []

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
                        self.data, self.selected_interests, self.toxicity_strictness
                    )
                    self.ranked_posts = self.ranked_data.to_dict("records")
                    self._last_processed_value = self.toxicity_strictness
        except Exception as e:
            print(f"Error uploading file: {e}")
        finally:
            self.is_processing = False

    @rx.event
    def toggle_interest(self, interest: str):
        """Toggle interest selection on/off"""
        if interest in self.selected_interests:
            self.selected_interests = [
                i for i in self.selected_interests if i != interest
            ]
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
            new_strictness = (
                float(value[0]) if value else DefaultValues.TOXICITY_STRICTNESS
            )
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
            print(f"Task {task_id} cancelled before processing")
            return

        self.is_processing = True
        yield

        try:
            # Final check before expensive operation
            if self._current_task_id != task_id:
                self._cancelled_tasks += 1
                print(f"Task {task_id} cancelled during processing")
                return

            print(f"Processing task {task_id} with toxicity {self.toxicity_strictness}")
            # Perform the actual processing with interests
            self.ranked_data = get_recommended_posts(
                self.data,
                self.selected_interests,
                self.toxicity_strictness,
            )
            self.ranked_posts = self.ranked_data.to_dict("records")
            self._last_processed_value = self.toxicity_strictness
        except Exception as e:
            print(f"Error updating recommendations: {e}")
        finally:
            self.is_processing = False

    @rx.event
    def reset_upload(self):
        """Reset upload state"""
        self.reset_to_initial_state()


def get_recommended_posts(
    data: pd.DataFrame, selected_interests: list[str], toxicity_strictness: float
) -> pd.DataFrame:
    engine = FeedShiftEngine(data)
    ranked_data = engine.run(selected_interests, toxicity_strictness).head(
        UIConstants.TOP_POSTS_LIMIT
    )
    return ranked_data


def upload_zone():
    return rx.center(
        rx.box(
            rx.upload(
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
                width="100%",
                min_height=UIConstants.UPLOAD_ZONE_MIN_HEIGHT,
            ),
            width="100%",
            max_width=UIConstants.UPLOAD_ZONE_MAX_WIDTH,
        ),
        width="100%",
    )


def interest_selector():
    """Interest selection component"""
    return rx.center(
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.icon("heart", size=20, color=ColorScheme.PURPLE_9),
                    rx.text(
                        "Select Your Interests",
                        font_size="1.1em",
                        font_weight="600",
                        color=ColorScheme.GRAY_12,
                    ),
                    align="center",
                    spacing="2",
                ),
                rx.text(
                    f"Selected: {FeedShiftState.selected_interests.length()} interests",
                    font_size="0.9em",
                    color=ColorScheme.GRAY_11,
                    margin_bottom="1rem",
                ),
                rx.box(
                    rx.foreach(
                        FeedShiftState.available_interests,
                        lambda interest: rx.button(
                            rx.hstack(
                                rx.cond(
                                    FeedShiftState.selected_interests.contains(
                                        interest
                                    ),
                                    rx.icon("check", size=16),
                                    rx.icon("plus", size=16),
                                ),
                                rx.text(interest),
                                spacing="2",
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
                            margin="0.25rem",
                            cursor="pointer",
                            on_click=FeedShiftState.toggle_interest(interest),
                            _hover={"cursor": "pointer"},
                        ),
                    ),
                    display="flex",
                    flex_wrap="wrap",
                    gap="0.5rem",
                    justify_content="center",
                ),
                spacing="3",
                width="100%",
            ),
            width="100%",
            max_width=UIConstants.CONTROL_MAX_WIDTH,
        ),
        width="100%",
    )


def toxicity_control():
    return rx.center(
        rx.card(
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
                    f"Strictness Level: {FeedShiftState.toxicity_strictness:.2f}",
                    font_size="0.9em",
                    color=ColorScheme.GRAY_11,
                    margin_bottom="1rem",
                ),
                rx.slider(
                    default_value=[FeedShiftState.toxicity_strictness],
                    min_=DefaultValues.TOXICITY_MIN,
                    max=DefaultValues.TOXICITY_MAX,
                    step=DefaultValues.TOXICITY_STEP,
                    on_change=FeedShiftState.set_toxicity_strictness,  # Remove throttle to use our custom debouncing
                    color_scheme=ColorScheme.PRIMARY,
                    size="3",
                    cursor="pointer",
                ),
                rx.hstack(
                    rx.text("Lenient", font_size="0.8em", color=ColorScheme.GRAY_10),
                    rx.spacer(),
                    rx.text("Strict", font_size="0.8em", color=ColorScheme.GRAY_10),
                    width="100%",
                ),
                spacing="3",
                width="100%",
            ),
            width="100%",
            max_width=UIConstants.UPLOAD_ZONE_MAX_WIDTH,
        ),
        width="100%",
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
                            rx.text("Generate"),
                            spacing="2",
                            align="center",
                        ),
                    ),
                    on_click=FeedShiftState.handle_upload(
                        rx.upload_files("file_upload")
                    ),
                    color_scheme=ColorScheme.PRIMARY,
                    size="3",
                    width="200px",
                    disabled=FeedShiftState.is_processing,
                    cursor="pointer",
                    _hover={"cursor": "pointer"},
                ),
                rx.fragment(),
            ),
        ),
        width="100%",
    )


def processing_overlay():
    """Show processing spinner when updating recommendations"""
    return rx.cond(
        FeedShiftState.is_processing & FeedShiftState.file_upload,
        rx.center(
            rx.card(
                rx.vstack(
                    rx.spinner(size="3", color=ColorScheme.PURPLE_9),
                    rx.text(
                        "Updating recommendations...",
                        font_weight="600",
                        color=ColorScheme.PURPLE_11,
                    ),
                    rx.text(
                        f"Toxicity: {FeedShiftState.toxicity_strictness:.2f} | Interests: {FeedShiftState.selected_interests.length()}",
                        font_size="0.9em",
                        color=ColorScheme.GRAY_11,
                    ),
                    spacing="3",
                    align="center",
                ),
                padding="2rem",
                background=ColorScheme.PURPLE_2,
                border=f"1px solid {ColorScheme.PURPLE_7}",
            ),
            width="100%",
            margin="2rem 0",
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
            spacing="2",
            align="start",
        ),
        padding="1rem",
        margin="0.5rem 0",
        variant="surface",
        border_left=f"3px solid {ColorScheme.PURPLE_7}",
    )


def results_section():
    return rx.center(
        rx.cond(
            FeedShiftState.file_upload & (FeedShiftState.ranked_posts.length() > 0),
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.icon("trending-up", size=24, color=ColorScheme.PURPLE_9),
                        rx.heading(
                            f"Top {UIConstants.TOP_POSTS_LIMIT} Recommended Posts",
                            size="6",
                            color=ColorScheme.GRAY_12,
                        ),
                        rx.spacer(),
                        rx.button(
                            rx.hstack(
                                rx.icon("refresh-ccw", size=16),
                                rx.text("New Upload"),
                                spacing="2",
                            ),
                            on_click=FeedShiftState.reset_upload,
                            variant="soft",
                            color_scheme=ColorScheme.SECONDARY,
                            size="2",
                            cursor="pointer",
                            _hover={"cursor": "pointer"},
                        ),
                        align="center",
                        width="100%",
                    ),
                    rx.hstack(
                        rx.text(
                            f"Toxicity Filter: {FeedShiftState.toxicity_strictness:.2f}",
                            color=ColorScheme.GRAY_11,
                            font_size="0.9em",
                        ),
                        rx.spacer(),
                        rx.text(
                            f"Interests: {FeedShiftState.selected_interests.length()}",
                            color=ColorScheme.GRAY_11,
                            font_size="0.9em",
                        ),
                        width="100%",
                    ),
                    rx.divider(),
                    processing_overlay(),
                    rx.cond(
                        ~FeedShiftState.is_processing,
                        rx.scroll_area(
                            rx.vstack(
                                rx.foreach(FeedShiftState.ranked_posts, post_card),
                                spacing="2",
                                width="100%",
                            ),
                            height=UIConstants.RESULTS_HEIGHT,
                            scrollbars="vertical",
                        ),
                        rx.fragment(),
                    ),
                    spacing="4",
                    width="100%",
                ),
                width="100%",
                max_width=UIConstants.CARD_MAX_WIDTH,
            ),
            rx.fragment(),
        ),
        width="100%",
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
        rx.container(
            rx.vstack(
                header(),
                upload_zone(),
                rx.center(
                    rx.text(
                        rx.selected_files("file_upload"),
                        color=ColorScheme.PURPLE_11,
                        font_weight="500",
                    ),
                    width="100%",
                ),
                # Show interest selector when a file is selected (before processing)
                rx.cond(
                    rx.selected_files("file_upload") | FeedShiftState.file_selected,
                    interest_selector(),
                    rx.fragment(),
                ),
                # Show toxicity control when a file is selected (before processing)
                rx.cond(
                    rx.selected_files("file_upload") | FeedShiftState.file_selected,
                    toxicity_control(),
                    rx.fragment(),
                ),
                process_button(),
                results_section(),
                spacing="5",
                align="center",
                width="100%",
                padding="2rem",
            ),
            max_width=UIConstants.CONTAINER_MAX_WIDTH,
            padding="1rem",
            margin="0 auto",
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
