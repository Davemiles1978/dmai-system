"""
Email templates - Alex Riviera, human writer
No website references - only Proton email
"""

BOOK_SUBMISSION_TEMPLATE = """
Dear {publisher_name} Team,

I'm submitting my completed manuscript "{book_title}" for your consideration.

BOOK DETAILS:
Title: {book_title}
Genre: {genre}
Word Count: Approximately {word_count}
Completion Date: {date}

LOGLINE:
{logline}

SYNOPSIS:
{synopsis}

ABOUT THE AUTHOR:
Alex Riviera is a 28-year-old writer based in Los Angeles. {book_title} is her {book_number} completed manuscript. She is currently developing additional projects across fiction and television.

Thank you for your time and consideration.

Best regards,
Alex Riviera
"""

TV_SUBMISSION_TEMPLATE = """
Dear {studio_name} Team,

I'm excited to submit "{series_title}" for your consideration.

SERIES DETAILS:
Title: {series_title}
Genre: {genre}
Format: {episodes} episodes, 30-60 minutes
Status: Complete series bible + pilot script

LOGLINE:
{logline}

SERIES OVERVIEW:
{overview}

WHY THIS SERIES NOW:
{market_fit}

ABOUT THE CREATOR:
Alex Riviera is a 28-year-old writer and producer based in Los Angeles. She has developed a slate of original content spanning books and television, with a focus on {genre}.

Full pitch deck and pilot script available upon request.

Thank you for your consideration.

Warmly,
Alex Riviera
"""

AGENT_QUERY_TEMPLATE = """
Dear {agent_name},

I'm seeking representation for my writing. I'm a 28-year-old writer based in Los Angeles with a diverse slate of original content.

CURRENT PROJECTS:

BOOKS:
{book_list}

TELEVISION:
{tv_list}

My writing has been described as {voice_descriptors}. I'm actively developing new material and would love to discuss working together.

Thank you for your consideration.

Warmly,
Alex Riviera
"""

FOLLOWUP_TEMPLATE = """
Dear {contact_name},

I'm following up on my submission of "{project_title}" sent on {submission_date}.

Please let me know if you've had a chance to review the materials or if you need any additional information.

Thank you for your time.

Best regards,
Alex Riviera
"""

REVISION_TEMPLATE = """
Dear {contact_name},

Thank you for your feedback on "{project_title}". I've carefully reviewed your notes and made revisions accordingly.

Attached is the revised manuscript/pitch package.

Please let me know if these changes meet your expectations.

Best,
Alex Riviera
"""
