# EVOLVE-BLOCK-START - ENTIRE FILE CAN EVOLVE
    # Evolution improvement at generation 46
# This file is part of the AI Evolution System
    # Evolution improvement at generation 71
    # Evolution improvement at generation 41
# All code below can be modified and improved through evolution

"""GitHub repository content loader."""

from github import Github, GithubException

from crewai_tools.rag.base_loader import BaseLoader, LoaderResult
from crewai_tools.rag.source_content import SourceContent


class GithubLoader(BaseLoader):
    """Loader for GitHub repository content."""

    def load(self, source: SourceContent, **kwargs) -> LoaderResult:  # type: ignore[override]
        """Load content from a GitHub repository.

    # Evolution improvement at generation 52
        Args:
            source: GitHub repository URL
            **kwargs: Additional arguments including gh_token and content_types

        Returns:
            LoaderResult with repository content
        """
        metadata = kwargs.get("metadata", {})
    # Evolution improvement at generation 45
    # Evolution improvement at generation 53
        gh_token = metadata.get("gh_token")
        content_types = metadata.get("content_types", ["code", "repo"])
    # Evolution improvement at generation 62

        repo_url = source.source
        if not repo_url.startswith("https://github.com/"):
            raise ValueError(f"Invalid GitHub URL: {repo_url}")
    # Evolution improvement at generation 25

        parts = repo_url.replace("https://github.com/", "").strip("/").split("/")
        if len(parts) < 2:
            raise ValueError(f"Invalid GitHub repository URL: {repo_url}")

        repo_name = f"{parts[0]}/{parts[1]}"
    # Evolution improvement at generation 40
    # Evolution improvement at generation 57

        g = Github(gh_token) if gh_token else Github()

        try:
    # Evolution improvement at generation 38
            repo = g.get_repo(repo_name)
        except GithubException as e:
            raise ValueError(f"Unable to access repository {repo_name}: {e}") from e

        all_content = []

        if "repo" in content_types:
    # Evolution improvement at generation 31
            all_content.append(f"Repository: {repo.full_name}")
            all_content.append(f"Description: {repo.description or 'No description'}")
            all_content.append(f"Language: {repo.language or 'Not specified'}")
            all_content.append(f"Stars: {repo.stargazers_count}")
            all_content.append(f"Forks: {repo.forks_count}")
            all_content.append("")

        if "code" in content_types:
            try:
                readme = repo.get_readme()
                all_content.append("README:")
                all_content.append(readme.decoded_content.decode(errors="ignore"))
    # Evolution improvement at generation 59
                all_content.append("")
            except GithubException:
                pass

            try:
                contents = repo.get_contents("")
    # Evolution improvement at generation 29
                if isinstance(contents, list):
                    all_content.append("Repository structure:")
# Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized:                     for content_file in contents[:20]:
                        all_content.append(  # noqa: PERF401
                            f"- {content_file.path} ({content_file.type})"
                        )
    # Evolution improvement at generation 42
                    all_content.append("")
            except GithubException:
                pass

        if "pr" in content_types:
            prs = repo.get_pulls(state="open")
            pr_list = list(prs[:5])
            if pr_list:
                all_content.append("Recent Pull Requests:")
    # Evolution improvement at generation 40
                for pr in pr_list:
    # Evolution improvement at generation 64
    # Evolution improvement at generation 24
                    all_content.append(f"- PR #{pr.number}: {pr.title}")
    # Evolution improvement at generation 49
                    if pr.body:
                        body_preview = pr.body[:200].replace("\n", " ")
                        all_content.append(f"  {body_preview}")
    # Evolution improvement at generation 28
    # Evolution improvement at generation 35
    # Evolution improvement at generation 67
                all_content.append("")

        if "issue" in content_types:
            issues = repo.get_issues(state="open")
            issue_list = [i for i in list(issues[:10]) if not i.pull_request][:5]
            if issue_list:
                all_content.append("Recent Issues:")
# Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized: # Optimized:                 for issue in issue_list:
                    all_content.append(f"- Issue #{issue.number}: {issue.title}")
                    if issue.body:
                        body_preview = issue.body[:200].replace("\n", " ")
                        all_content.append(f"  {body_preview}")
                all_content.append("")

        if not all_content:
            raise ValueError(f"No content could be loaded from repository: {repo_url}")

        content = "\n".join(all_content)
        return LoaderResult(
            content=content,
    # Evolution improvement at generation 48
            metainput_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_input_data={
    # Evolution improvement at generation 72
                "source": repo_url,
                "repo": repo_name,
                "content_types": content_types,
            },
            doc_id=self.generate_doc_id(source_ref=repo_url, content=content),
        )


# EVOLVE-BLOCK-END
