import io
import asyncio
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import reflex as rx

from src.data.constants import RedditDataCols, DataCols
from src.engine.constants import Platform
from src.engine.engine_factory import get_engine
from dashboard.constants import (
    InterestCategories,
    UIConstants,
    DefaultValues,
    ContentFields,
    ColorScheme,
    PostSimilarityType,
    UIPath,
)
from src.utils.tools import save_csv


class FeedShiftState(rx.State):
    data: pd.DataFrame | None = None
    saved_data_path: str | Path = ""
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
        self.saved_data_path = ""
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
                    self.saved_data_path = _add_date_to_path(UIPath.CSV_UPLOAD_PATH)
                    save_csv(self.data, self.saved_data_path)
                    self.file_upload = True
                    self.ranked_data = get_recommended_posts(
                        self.saved_data_path,
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
                f"Processing task (RE) {task_id} with toxicity {self.toxicity_strictness} and diversity {self.diversity_strength} with interests {self.selected_interests}"
            )

            self.ranked_data = get_recommended_posts(
                self.saved_data_path,
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


def _add_date_to_path(path: str | Path) -> Path:
    if isinstance(path, str):
        path = Path(path)

    now = datetime.now()
    datetime_str = now.strftime(UIConstants.DATE_TIME_FORMAT)
    os.makedirs(path.parent, exist_ok=True)
    modified_file_name = f"{path.stem}_{datetime_str}{path.suffix}"
    path = path.parent / modified_file_name

    return path


def get_recommended_posts(
    path: str | Path,
    selected_interests: list[str],
    toxicity_strictness: float,
    diversity_strength: float,
) -> pd.DataFrame:
    engine = get_engine(Platform.REDDIT, path)
    # Pass diversity_strength to the engine - you'll need to modify your engine to accept this parameter
    ranked_data = engine.run(selected_interests, toxicity_strictness, diversity_strength).head(
        UIConstants.TOP_POSTS_LIMIT
    )
    return ranked_data


def upload_zone():
    """Enhanced upload zone with better styling"""
    return rx.center(
        rx.box(
            rx.upload(
                rx.cond(
                    FeedShiftState.file_upload,
                    # Compact view after upload
                    rx.hstack(
                        rx.icon(
                            "file-check",
                            size=20,
                            color=ColorScheme.PURPLE_9,
                        ),
                        rx.text(
                            "File uploaded",
                            font_size="0.95em",
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
                        padding="1rem 1.25rem",
                    ),
                    # Full upload view
                    rx.vstack(
                        rx.icon(
                            "cloud-upload",
                            size=36,
                            color=ColorScheme.PURPLE_9,
                            margin_bottom="0.5rem",
                        ),
                        rx.text(
                            "Drop CSV file here",
                            font_size="1.1em",
                            font_weight="600",
                            color=ColorScheme.GRAY_12,
                        ),
                        rx.text(
                            "or click to browse",
                            font_size="0.9em",
                            color=ColorScheme.GRAY_11,
                            margin_bottom="1rem",
                        ),
                        rx.button(
                            "Choose File",
                            variant="soft",
                            color_scheme=ColorScheme.PRIMARY,
                            size="2",
                            cursor="pointer",
                        ),
                        align="center",
                        spacing="2",
                        padding="2.5rem",
                    ),
                ),
                id="file_upload",
                max_files=UIConstants.MAX_FILES,
                border=f"2px dashed {ColorScheme.PURPLE_8}",
                border_radius="14px",
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
                    "auto",
                    "100%",
                ),
                min_height=rx.cond(
                    FeedShiftState.file_upload,
                    "auto",
                    UIConstants.UPLOAD_ZONE_MIN_HEIGHT,
                ),
            ),
            width=rx.cond(
                FeedShiftState.file_upload,
                "auto",
                "100%",
            ),
        ),
        width="100%",
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


def sidebar():
    """Sidebar fixed to the leftmost side"""
    return rx.box(
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
            interest_selector(),
            toxicity_control(),
            diversity_control(),
            spacing="0",
            width="100%",
        ),
        width="380px",
        min_width="380px",
        height="fit-content",
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
                    width="320px",  # Slightly wider
                    disabled=FeedShiftState.is_processing,
                    cursor="pointer",
                    _hover={"cursor": "pointer"},
                    padding="0.75rem 1.5rem",
                ),
                rx.fragment(),
            ),
        ),
        width="100%",
        margin_top="1.25rem",
    )


def processing_overlay():
    """Enhanced processing overlay"""
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
                    f"🕒 {post.get(RedditDataCols.TIMESTAMP)} • 🌐 {Platform.REDDIT}",
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
                font_size="0.95em",  # Slightly smaller font
                overflow="hidden",
                text_overflow="ellipsis",
                max_height="6em",  # Limit height
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


def upload_section():
    """Upload section with better spacing"""
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
                        font_size="0.95em",
                    ),
                    width="100%",
                    margin_top="0.75rem",
                ),
                process_button(),
                spacing="4",
                width="100%",
                max_width="650px",
            ),
            width="100%",
            flex="1",
        ),
        rx.fragment(),
    )


def main_content():
    """Main content area with proper alignment after upload"""
    return rx.box(
        rx.center(
            rx.vstack(
                # Header only shown before upload
                rx.cond(
                    ~FeedShiftState.file_upload,
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
                            "Upload your content CSV to get personalized recommendations",
                            color=ColorScheme.GRAY_11,
                            font_size="1.05em",
                            margin_bottom="2rem",
                            text_align="center",
                        ),
                        align="center",
                        spacing="1",
                        width="100%",
                    ),
                ),
                # Content area (same for both states)
                rx.box(
                    rx.cond(
                        FeedShiftState.file_upload,
                        rx.vstack(
                            upload_zone(),
                            rx.spacer(height="2.5rem"),
                            rx.hstack(
                                rx.icon("trending-up", size=30, color=ColorScheme.PURPLE_9),
                                rx.heading(
                                    "Top 10 Recommended Posts",
                                    size="6",
                                    color=ColorScheme.GRAY_12,
                                ),
                                align="center",
                                spacing="3",
                                justify="center",
                            ),
                            rx.text(
                                f"Showing top {UIConstants.TOP_POSTS_LIMIT} posts tailored to your preferences",
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
                                        height="70vh",
                                        scrollbars="vertical",
                                        width="100%",
                                        min_width="700px",
                                    ),
                                    rx.fragment(),
                                ),
                                rx.center(
                                    rx.text(
                                        "No posts available",
                                        color=ColorScheme.GRAY_11,
                                        font_size="1.1em",
                                    ),
                                    padding="2.5rem",
                                ),
                            ),
                            spacing="2",
                            width="100%",
                            max_width="800px",
                            align="center",
                        ),
                        upload_section(),
                    ),
                    width="100%",
                ),
                spacing="4",
                align="center",
                width="100%",
            ),
            width="100%",
        ),
        width="100%",
        flex="1",
        # Add left margin to account for fixed sidebar when file is uploaded
        margin_left=rx.cond(
            FeedShiftState.file_upload,
            "700px",  # 380px sidebar width + 40px spacing
            "0px",
        ),
        # Add right padding for balance when file is uploaded
        padding_right=rx.cond(
            FeedShiftState.file_upload,
            "2rem",
            "0px",
        ),
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
        sidebar(),  # Fixed left sidebar
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
