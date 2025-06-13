import io
import time
import asyncio
from typing import Optional

import pandas as pd
import reflex as rx

from src.engine.run_engine import FeedShiftEngine


class FeedShiftState(rx.State):
    data: pd.DataFrame | None = None
    uploaded_content: str = ""
    toxicity_strictness: float = 0.5
    ranked_data: pd.DataFrame | None = None
    ranked_posts: list[dict] = []
    file_upload: bool = False
    is_processing: bool = False
    _current_task_id: int = 0  # Simple counter for task identification
    _last_processed_value: float = 0.5  # Track the last processed value

    def reset_to_initial_state(self):
        """Reset all state to initial values"""
        self.data = None
        self.uploaded_content = ""
        self.toxicity_strictness = 0.5
        self.ranked_data = None
        self.ranked_posts = []
        self.file_upload = False
        self.is_processing = False
        self._current_task_id = 0
        self._last_processed_value = 0.5

    @rx.event
    async def handle_upload(self, files: list[rx.UploadFile]):
        if not files:
            return

        self.is_processing = True
        yield

        try:
            file = files[0]
            if file.name.endswith('.csv'):
                content = await file.read()
                self.data = pd.read_csv(io.StringIO(content.decode('utf-8')))
                self.uploaded_content = self.data.head().to_markdown()

                if self.data is not None:
                    self.file_upload = True
                    self.ranked_data = get_recommended_posts(self.data, self.toxicity_strictness)
                    self.ranked_posts = self.ranked_data.to_dict('records')
                    self._last_processed_value = self.toxicity_strictness
        except Exception as e:
            print(f"Error uploading file: {e}")
        finally:
            self.is_processing = False

    @rx.event
    def set_toxicity_strictness(self, value):
        if isinstance(value, list):
            new_strictness = float(value[0]) if value else 0.5
        else:
            new_strictness = float(value)

        self.toxicity_strictness = new_strictness

        # Only process if we have data and file is uploaded
        if self.data is not None and self.file_upload:
            # Increment task ID to invalidate previous tasks
            self._current_task_id += 1
            current_task = self._current_task_id

            # Schedule the update with the current task ID
            return FeedShiftState.update_recommendations_delayed(current_task)

    @rx.event
    async def update_recommendations_delayed(self, task_id: int):
        """Update recommendations with proper debouncing using task ID"""
        # Wait for debounce period
        await asyncio.sleep(0.3)  # 300ms debounce

        # Check if this task is still the current one
        if self._current_task_id != task_id:
            return  # A newer request came in, abort this one

        # Check if the value has already been processed
        if abs(self._last_processed_value - self.toxicity_strictness) < 0.001:
            return  # Same value already processed

        # Check if we still have data and should process
        if self.data is None or not self.file_upload:
            return

        self.is_processing = True
        yield

        try:
            # Perform the actual processing
            self.ranked_data = get_recommended_posts(self.data, self.toxicity_strictness)
            self.ranked_posts = self.ranked_data.to_dict('records')
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
        data: pd.DataFrame, toxicity_strictness: float
) -> pd.DataFrame:
    engine = FeedShiftEngine(data)
    ranked_data = engine.run(toxicity_strictness).head(10)
    return ranked_data


def upload_zone():
    return rx.center(
        rx.box(
            rx.upload(
                rx.vstack(
                    rx.icon(
                        "cloud-upload",
                        size=32,
                        color="var(--purple-9)",
                    ),
                    rx.text(
                        "Drop CSV file here",
                        font_size="1em",
                        font_weight="600",
                        color="var(--gray-12)",
                    ),
                    rx.text(
                        "or click to browse",
                        font_size="0.8em",
                        color="var(--gray-11)",
                    ),
                    rx.button(
                        "Choose File",
                        variant="soft",
                        color_scheme="purple",
                        size="2",
                        margin_top="0.5rem",
                        cursor="pointer",
                    ),
                    align="center",
                    spacing="2",
                    padding="2rem",
                ),
                id="file_upload",
                max_files=1,
                border="2px dashed var(--purple-8)",
                border_radius="12px",
                background="var(--purple-2)",
                transition="all 0.2s ease",
                cursor="pointer",
                _hover={
                    "border_color": "var(--purple-9)",
                    "background": "var(--purple-3)",
                    "cursor": "pointer",
                },
                width="100%",
                min_height="150px",
            ),
            width="100%",
            max_width="400px",
        ),
        width="100%",
    )


def toxicity_control():
    return rx.center(
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.icon("shield-check", size=20, color="var(--purple-9)"),
                    rx.text(
                        "Toxicity Filter",
                        font_size="1.1em",
                        font_weight="600",
                        color="var(--gray-12)",
                    ),
                    align="center",
                    spacing="2",
                ),
                rx.text(
                    f"Strictness Level: {FeedShiftState.toxicity_strictness:.2f}",
                    font_size="0.9em",
                    color="var(--gray-11)",
                    margin_bottom="1rem",
                ),
                rx.slider(
                    default_value=[FeedShiftState.toxicity_strictness],
                    min_=0,
                    max=1,
                    step=0.05,
                    on_change=FeedShiftState.set_toxicity_strictness.throttle(10),
                    # Reduced throttle for better responsiveness
                    color_scheme="purple",
                    size="3",
                    cursor="pointer",
                ),
                rx.hstack(
                    rx.text("Lenient", font_size="0.8em", color="var(--gray-10)"),
                    rx.spacer(),
                    rx.text("Strict", font_size="0.8em", color="var(--gray-10)"),
                    width="100%",
                ),
                spacing="3",
                width="100%",
            ),
            width="100%",
            max_width="400px",
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
                    on_click=FeedShiftState.handle_upload(rx.upload_files("file_upload")),
                    color_scheme="purple",
                    size="3",
                    width="200px",
                    disabled=FeedShiftState.is_processing,
                    cursor="pointer",
                    _hover={"cursor": "pointer"},
                ),
                rx.fragment(),
            )
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
                    rx.spinner(size="3", color="var(--purple-9)"),
                    rx.text(
                        "Updating recommendations...",
                        font_weight="600",
                        color="var(--purple-11)",
                    ),
                    rx.text(
                        f"Toxicity Level: {FeedShiftState.toxicity_strictness:.2f}",
                        font_size="0.9em",
                        color="var(--gray-11)",
                    ),
                    spacing="3",
                    align="center",
                ),
                padding="2rem",
                background="var(--purple-2)",
                border="1px solid var(--purple-7)",
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
                    color_scheme="purple",
                    variant="soft",
                ),
                rx.spacer(),
                rx.text(
                    rx.cond(
                        post.get("timestamp"),
                        post.get("timestamp"),
                        rx.cond(
                            post.get("created_at"),
                            post.get("created_at"),
                            "No timestamp"
                        )
                    ),
                    color="var(--gray-11)",
                    font_size="0.9em",
                ),
                width="100%",
                align="center",
            ),
            rx.text(
                rx.cond(
                    post.get("content"),
                    post.get("content"),
                    rx.cond(
                        post.get("text"),
                        post.get("text"),
                        rx.cond(
                            post.get("body"),
                            post.get("body"),
                            "No content available"
                        )
                    )
                ),
                color="var(--gray-12)",
                line_height="1.5",
            ),
            rx.hstack(
                rx.text(
                    rx.cond(
                        post.get("score"),
                        f"Score: {post.get('score')}",
                        rx.cond(
                            post.get("ranking_score"),
                            f"Score: {post.get('ranking_score')}",
                            "Score: N/A"
                        )
                    ),
                    font_weight="600",
                    color="var(--purple-11)",
                    font_size="0.9em",
                ),
                rx.spacer(),
                rx.text(
                    rx.cond(
                        post.get("author"),
                        f"Author: {post.get('author')}",
                        rx.cond(
                            post.get("username"),
                            f"Author: {post.get('username')}",
                            rx.cond(
                                post.get("user"),
                                f"Author: {post.get('user')}",
                                "Author: Unknown"
                            )
                        )
                    ),
                    color="var(--gray-10)",
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
        border_left="3px solid var(--purple-7)",
    )


def results_section():
    return rx.center(
        rx.cond(
            FeedShiftState.file_upload & (FeedShiftState.ranked_posts.length() > 0),
            rx.card(
                rx.vstack(
                    rx.hstack(
                        rx.icon("trending-up", size=24, color="var(--purple-9)"),
                        rx.heading(
                            "Top 10 Recommended Posts",
                            size="6",
                            color="var(--gray-12)",
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
                            color_scheme="gray",
                            size="2",
                            cursor="pointer",
                            _hover={"cursor": "pointer"},
                        ),
                        align="center",
                        width="100%",
                    ),
                    rx.text(
                        f"Toxicity Filter: {FeedShiftState.toxicity_strictness:.2f}",
                        color="var(--gray-11)",
                        font_size="0.9em",
                    ),
                    rx.divider(),
                    processing_overlay(),
                    rx.cond(
                        ~FeedShiftState.is_processing,
                        rx.scroll_area(
                            rx.vstack(
                                rx.foreach(
                                    FeedShiftState.ranked_posts,
                                    post_card
                                ),
                                spacing="2",
                                width="100%",
                            ),
                            height="400px",
                            scrollbars="vertical",
                        ),
                        rx.fragment(),
                    ),
                    spacing="4",
                    width="100%",
                ),
                width="100%",
                max_width="800px",
            ),
            rx.fragment(),
        ),
        width="100%",
    )


def header():
    return rx.center(
        rx.vstack(
            rx.image(
                src="feedshift_main_logo-removebg.png",
                alt="Feedshift Logo",
                height="250px",
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
                        color="var(--purple-11)",
                        font_weight="500",
                    ),
                    width="100%",
                ),
                toxicity_control(),
                process_button(),
                results_section(),
                spacing="5",
                align="center",
                width="100%",
                padding="2rem",
            ),
            max_width="1200px",
            padding="1rem",
            margin="0 auto",
        ),
        background="radial-gradient(ellipse at top, var(--purple-3), transparent), radial-gradient(ellipse at bottom, var(--purple-2), transparent)",
        min_height="100vh",
        width="100%",
    )


app = rx.App(
    theme=rx.theme(
        appearance="dark",
        has_background=True,
        radius="large",
        accent_color="purple",
    )
)
app.add_page(index)