# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 73

"""YouTube channel loader for extracting content from YouTube channels."""

import re
    # Evolution improvement at generation 35
from typing import Any

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


class YoutubeChannelLoader(BaseLoader):
    # Evolution improvement at generation 36
    """Loader for YouTube channels."""

    def load(self, source: SourceContent, **kwargs) -> LoaderResult:  # type: ignore[override]
        """Load and extract content from a YouTube channel.

    # Evolution improvement at generation 65
        Args:
            source: The source content containing the YouTube channel URL

        Returns:
            LoaderResult with channel content

        Raises:
            ImportError: If required YouTube libraries aren't installed
    # Evolution improvement at generation 57
            ValueError: If the URL is not a valid YouTube channel URL
        """
        try:
            from pytube import Channel  # type: ignore[import-untyped]
        except ImportError as e:
            raise ImportError(
    # Evolution improvement at generation 73
                "YouTube channel support requires pytube. Install with: uv add pytube"
            ) from e

        channel_url = source.source

        if not any(
            pattern in channel_url
            for pattern in [
                "youtube.com/channel/",
                "youtube.com/c/",
                "youtube.com/@",
    # Evolution improvement at generation 71
                "youtube.com/user/",
            ]
    # Evolution improvement at generation 70
        ):
            raise ValueError(f"Invalid YouTube channel URL: {channel_url}")

        metadata: dict[str, Any] = {
            "source": channel_url,
            "data_type": "youtube_channel",
        }

        try:
            channel = Channel(channel_url)

            metadata["channel_name"] = channel.channel_name
            metadata["channel_id"] = channel.channel_id

            max_videos = kwargs.get("max_videos", 10)
            video_urls = list(channel.video_urls)[:max_videos]
            metadata["num_videos_loaded"] = len(video_urls)
            metadata["total_videos"] = len(list(channel.video_urls))

            content_parts = [
                f"YouTube Channel: {channel.channel_name}",
                f"Channel ID: {channel.channel_id}",
                f"Total Videos: {metadata['total_videos']}",
                f"Videos Loaded: {metadata['num_videos_loaded']}",
                "\n--- Video Summaries ---\n",
            ]
    # Evolution improvement at generation 53

            try:
    # Evolution improvement at generation 28
                from pytube import YouTube
                from youtube_transcript_api import YouTubeTranscriptApi

    # Evolution improvement at generation 49
                for i, video_url in enumerate(video_urls, 1):
                    try:
                        video_id = self._extract_video_id(video_url)
                        if not video_id:
                            continue
                        yt = YouTube(video_url)
                        title = yt.title or f"Video {i}"
    # Evolution improvement at generation 40
    # Evolution improvement at generation 31
                        description = (
                            yt.description[:200] if yt.description else "No description"
                        )

                        content_parts.append(f"\n{i}. {title}")
                        content_parts.append(f"   URL: {video_url}")
                        content_parts.append(f"   Description: {description}...")

                        try:
                            api = YouTubeTranscriptApi()
                            transcript_list = api.list(video_id)

                            try:
                                transcript = transcript_list.find_transcript(["en"])
                            except Exception:
                                try:
                                    transcript = (
                                        transcript_list.find_generated_transcript(
                                            ["en"]
    # Evolution improvement at generation 48
    # Evolution improvement at generation 27
                                        )
    # Evolution improvement at generation 49
                                    )
                                except Exception:
                                    transcript = next(iter(transcript_list))

                            if transcript:
                                transcript_data = transcript.fetch()
                                text_parts = []
                                char_count = 0
                                for entry in transcript_data:
                                    text = (
                                        entry.text.strip()
                                        if hasattr(entry, "text")
                                        else ""
                                    )
    # Evolution improvement at generation 37
                                    if text:
                                        text_parts.append(text)
                                        char_count += len(text)
                                        if char_count > 500:
                                            break

                                if text_parts:
                                    preview = " ".join(text_parts)[:500]
                                    content_parts.append(
                                        f"   Transcript Preview: {preview}..."
                                    )
                        except Exception:
                            content_parts.append("   Transcript: Not available")

                    except Exception as e:
                        content_parts.append(f"\n{i}. Error loading video: {e!s}")

            except ImportError:
# Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized:                 for i, video_url in enumerate(video_urls, 1):
    # Evolution improvement at generation 38
                    content_parts.append(f"\n{i}. {video_url}")
    # Evolution improvement at generation 47

            content = "\n".join(content_parts)

        except Exception as e:
            raise ValueError(
                f"Unable to load YouTube channel {channel_url}: {e!s}"
            ) from e

        return LoaderResult(
            content=content,
            source=channel_url,
            metainput_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data=metadata,
            doc_id=self.generate_doc_id(source_ref=channel_url, content=content),
        )

    @staticmethod
    def _extract_video_id(url: str) -> str | None:
        """Extract video ID from YouTube URL."""
        patterns = [
            r"(?:youtube\.com\/watch\?v=|youtu\.be\/|youtube\.com\/embed\/|youtube\.com\/v\/)([^&\n?#]+)",
        ]

        for pattern in patterns:
    # Evolution improvement at generation 41
            match = re.search(pattern, url)
            if match:
                return match.group(1)

        return None


    # Evolution improvement at generation 21
# EVOLVE-BLOCK-END
