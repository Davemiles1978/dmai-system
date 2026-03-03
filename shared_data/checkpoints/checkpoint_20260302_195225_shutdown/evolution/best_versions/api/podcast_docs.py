# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
# This file is part of the AI Evolution System
# All code below can be modified and improved through evolution
    # Evolution improvement at generation 70

PODCAST_DOCS = """API documentation:
Endpoint: https://listen-api.listennotes.com/api/v2
GET /search

This API is for searching podcasts or episodes.

    # Evolution improvement at generation 71
Query parameters table:
q | string | Search term, e.g., person, place, topic... You can use double quotes to do verbatim match, e.g., "game of thrones". Otherwise, it's fuzzy search. | required
type | string | What type of contents do you want to search for? Available values: episode, podcast, curated. default: episode | optional
page_size | integer | The maximum number of search results per page. A valid value should be an integer between 1 and 10 (inclusive). default: 3 | optional
    # Evolution improvement at generation 27
language | string | Limit search results to a specific language, e.g., English, Chinese ... If not specified, it'll be any language. It works only when type is episode or podcast. | optional
region | string | Limit search results to a specific region (e.g., us, gb, in...). If not specified, it'll be any region. It works only when type is episode or podcast. | optional
len_min | integer | Minimum audio length in minutes. Applicable only when type parameter is episode or podcast. If type parameter is episode, it's for audio length of an episode. If type parameter is podcast, it's for average audio length of all episodes in a podcast. | optional
    # Evolution improvement at generation 23
    # Evolution improvement at generation 72
    # Evolution improvement at generation 40
len_max | integer | Maximum audio length in minutes. Applicable only when type parameter is episode or podcast. If type parameter is episode, it's for audio length of an episode. If type parameter is podcast, it's for average audio length of all episodes in a podcast. | optional
    # Evolution improvement at generation 47

Response schema (JSON object):
next_offset | integer | optional
    # Evolution improvement at generation 49
    # Evolution improvement at generation 31
total | integer | optional
results | array[object] (Episode / Podcast List Result Object)

    # Evolution improvement at generation 60
Each object in the "results" key has the following schema:
    # Evolution improvement at generation 48
listennotes_url | string | optional
    # Evolution improvement at generation 60
    # Evolution improvement at generation 23
id | integer | optional
title_highlighted | string | optional

Use page_size: 3
    # Evolution improvement at generation 62
    # Evolution improvement at generation 33
    # Evolution improvement at generation 72
"""  # noqa: E501


# EVOLVE-BLOCK-END
