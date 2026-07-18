# bookbot

BookBot started as a [Boot.dev](https://www.boot.dev) project, and I am expanding the project to incorporate more text data processing techniques, including with the [LLooM](https://stanfordhci.github.io/lloom/about/) research tool for data analysis on unstructured text data.

# tips from install

- Expect the `mistralai` and `langchain` dependencies to be fussy with versions. To fix version conflicts, try installing the conflicting packages in the same install command (this forces `pip` to resolve conflicts at install rather than by bumping on them during subsequent installs).

# current version requirements

- python>=3.9,<3.13
- numpy<=2.0
