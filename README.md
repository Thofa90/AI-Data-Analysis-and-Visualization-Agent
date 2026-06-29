# AI Data Analysis and Visualization Agent

## Project Headline

An AI-powered, local-first data analysis and visualization assistant built with
Streamlit, Pandas, Plotly, and Ollama. The application helps users upload
business datasets, inspect data quality, clean data, ask natural-language
questions, generate charts, save insights, and evaluate answer quality without
writing code.

## Project Goal

The goal of this project is to make exploratory data analysis easier for users
who understand their business data but may not know Python, Pandas, SQL, or
visualization libraries.

Instead of forcing users to manually write formulas or scripts, the app lets
them ask questions such as:

- Which region has the highest sales?
- Show monthly profit for 2021.
- Which sub-category has the lowest profit?
- What percentage of revenue comes from each region?
- Create a scatter plot of Sales versus Profit.
- Tell me about this dataset.

The system then converts the request into safe, validated analytical operations,
performs the calculation using Pandas/SciPy, creates Plotly charts, and explains
the result in plain language.

## Business Application

This project can be used as a business intelligence assistant for teams that
work with CSV or Excel datasets.

Possible business use cases include:

- sales and revenue analysis
- profit and loss analysis
- regional performance comparison
- product, category, and sub-category performance tracking
- customer and order analysis
- inventory and quantity analysis
- marketing campaign analysis
- HR and employee analytics
- operational KPI tracking
- quick executive reporting
- data-quality review before dashboard creation

For example, a sales manager can upload an order dataset and ask:

```text
Which ship mode generated the highest total sales?
Show total revenue and total profit by year.
Which states have negative profit?
What percentage of total revenue comes from each region?
```

The app returns verified calculations, result tables, and charts that can be
used for reporting or decision support.

## Benefits of an AI Agent for Data Analysis and Visualization

- **Faster analysis:** Users can ask questions directly instead of writing code.
- **Lower technical barrier:** Non-technical users can explore datasets with
  natural language.
- **Verified calculations:** The LLM does not calculate numbers directly.
  Calculations are performed by deterministic Pandas/SciPy tools.
- **Automatic chart generation:** The app can produce bar charts, line charts,
  scatter plots, histograms, box plots, pie charts, heatmaps, treemaps, maps,
  bullet charts, and other supported visuals.
- **Dataset-aware context:** The app profiles uploaded data and uses actual
  column names, data types, and values to understand questions.
- **Safer execution:** The agent uses an allowlisted tool registry instead of
  arbitrary code execution.
- **Business-friendly explanations:** Results are translated into readable
  summaries, key findings, evidence, and next steps.
- **Data-quality support:** Users can inspect missing values, duplicates,
  data types, outliers, and possible cleaning actions.
- **Evaluation support:** Users can evaluate whether a chat answer is relevant,
  faithful to the result, complete, and supported by the selected tool output.
- **Local-first AI:** Ollama can run locally, reducing dependency on external
  cloud services for model calls.

## High-Level Application Flow

```text
User opens Streamlit app
        |
        v
app.py initializes settings, session state, styles, and navigation
        |
        v
User uploads CSV / XLSX / XLS file
        |
        v
services/file_loader.py validates and loads the dataset
        |
        v
services/dataset_profiler.py detects schema, types, dates, IDs, quality issues
        |
        v
Pages display overview, EDA, cleaning, visualization, chat, reports, evaluation
        |
        v
User asks a natural-language question in chat
        |
        v
agent/data_agent.py creates a safe analytical plan
        |
        v
agent/tool_registry.py validates the selected tool
        |
        v
agent/tools.py performs the Pandas/SciPy calculation
        |
        v
services/chart_service.py creates the Plotly chart when needed
        |
        v
services/llm_service.py / Ollama optionally explains the verified result
        |
        v
pages/chat_page.py renders answer, table, chart, insight, export, evaluation
```

## Python File Flow Diagram

The project is organized so each file has a focused responsibility.

```text
app.py
|
|-- config/
|   |-- settings.py
|   |   Loads environment variables and runtime app settings.
|   |
|   |-- __init__.py
|       Marks config as a Python package.
|
|-- utils/
|   |-- session_state.py
|   |   Initializes and resets Streamlit session state.
|   |
|   |-- navigation.py
|   |   Defines page navigation and route labels.
|   |
|   |-- logging_config.py
|   |   Configures application logging.
|   |
|   |-- formatting.py
|   |   Formats numbers, currency, percentages, dates, and display text.
|   |
|   |-- chat_text.py
|   |   Cleans and renders chat markdown safely.
|   |
|   |-- __init__.py
|       Marks utils as a Python package.
|
|-- components/
|   |-- sidebar.py
|   |   Renders sidebar navigation and dataset controls.
|   |
|   |-- key_metrics.py
|   |   Displays high-level metric cards.
|   |
|   |-- metric_cards.py
|   |   Renders reusable metric-card UI blocks.
|   |
|   |-- quality_card.py
|   |   Displays data-quality score and quality warnings.
|   |
|   |-- status_cards.py
|   |   Displays status summaries for app and data state.
|   |
|   |-- chart_panel.py
|   |   Renders charts and chart insight panels.
|   |
|   |-- chat_narrative.py
|   |   Renders structured chat explanations.
|   |
|   |-- error_message.py
|   |   Displays user-friendly error messages.
|   |
|   |-- __init__.py
|       Marks components as a Python package.
|
|-- pages/
|   |-- home_page.py
|   |   Landing/home page for file upload and project entry.
|   |
|   |-- overview_page.py
|   |   Dataset overview, schema summary, and quality summary.
|   |
|   |-- eda_page.py
|   |   Automatic exploratory data analysis.
|   |
|   |-- visualization_page.py
|   |   Manual chart builder and visualization controls.
|   |
|   |-- chat_page.py
|   |   Natural-language analytics chat interface.
|   |
|   |-- cleaning_page.py
|   |   Data-cleaning preview, confirmation, undo, and reset.
|   |
|   |-- correlation_page.py
|   |   Correlation analysis and relationship views.
|   |
|   |-- reports_page.py
|   |   Report generation and export workflow.
|   |
|   |-- saved_insights_page.py
|   |   Saved chat/chart insights.
|   |
|   |-- evaluation_page.py
|   |   Answer evaluation and benchmark dashboard.
|   |
|   |-- settings_page.py
|   |   Runtime configuration and model settings UI.
|   |
|   |-- __init__.py
|       Marks pages as a Python package.
|
|-- services/
|   |-- file_loader.py
|   |   Validates and loads CSV, XLSX, and XLS files.
|   |
|   |-- profile_models.py
|   |   Pydantic/data models for dataset profiling output.
|   |
|   |-- dataset_profiler.py
|   |   Detects columns, data types, dates, IDs, missing values, duplicates,
|   |   cardinality, outliers, and quality score.
|   |
|   |-- metric_detector.py
|   |   Detects likely business metrics and useful numeric columns.
|   |
|   |-- eda_service.py
|   |   Generates automated exploratory summaries and EDA observations.
|   |
|   |-- chart_service.py
|   |   Validates chart specifications and builds Plotly figures.
|   |
|   |-- chart_insight_service.py
|   |   Creates written insights from chart result data.
|   |
|   |-- date_aggregation_service.py
|   |   Handles year, quarter, month, week, and day date aggregation.
|   |
|   |-- cleaning_service.py
|   |   Provides safe data-cleaning recommendations and transformations.
|   |
|   |-- query_guide.py
|   |   Creates dataset-aware question guidance and examples.
|   |
|   |-- export_service.py
|   |   Exports data, charts, tables, and reports.
|   |
|   |-- report_service.py
|   |   Builds report content from analysis results.
|   |
|   |-- evaluation_service.py
|   |   Coordinates answer evaluation from deterministic and LLM evidence.
|   |
|   |-- llm_service.py
|   |   Connects the app to Ollama/LangChain planning and explanation.
|   |
|   |-- ollama_service.py
|   |   Checks Ollama availability and model status.
|   |
|   |-- __init__.py
|       Marks services as a Python package.
|
|-- agent/
|   |-- data_agent.py
|   |   Main chat brain. Converts user questions into safe analytical plans,
|   |   handles deterministic parsing, optional LLM planning, result explanation,
|   |   chart metadata, and response assembly.
|   |
|   |-- tools.py
|   |   Approved Pandas/SciPy analysis tools. Performs actual calculations.
|   |
|   |-- tool_registry.py
|   |   Allowlist of tools the agent is permitted to execute.
|   |
|   |-- schemas.py
|   |   Pydantic models for plans, responses, tool results, and chat outputs.
|   |
|   |-- prompts.py
|   |   LLM prompt templates and instructions.
|   |
|   |-- memory.py
|   |   Short session memory for follow-up questions.
|   |
|   |-- __init__.py
|       Marks agent as a Python package.
|
|-- evaluation/
|   |-- deterministic_metrics.py
|   |   Rule-based scoring for correctness, relevancy, completeness, faithfulness,
|   |   chart accuracy, and error handling.
|   |
|   |-- llm_judge.py
|   |   Optional Ollama-based qualitative answer judge.
|   |
|   |-- judge_prompts.py
|   |   Prompts used by the LLM judge.
|   |
|   |-- benchmark_runner.py
|   |   Runs benchmark test cases against sample datasets.
|   |
|   |-- evaluation_models.py
|   |   Pydantic models for evaluation results.
|   |
|   |-- __init__.py
|       Marks evaluation as a Python package.
|
|-- tests/
    |-- test_*.py
        Unit and behavior tests for loading, profiling, planning, tools,
        charts, formatting, cleaning, evaluation, and Streamlit page logic.
```

## Agent Execution Flow

```text
Question from user
        |
        v
agent/data_agent.py
 - reads dataset profile
 - detects columns, metrics, filters, dates, ranking words, chart intent
 - builds AgentPlan
        |
        v
agent/tool_registry.py
 - checks that requested tool is allowlisted
 - validates tool arguments
        |
        v
agent/tools.py
 - performs calculation using Pandas/SciPy
 - returns ToolResult with verified data
        |
        v
services/chart_service.py
 - creates chart if the plan includes ChartSpec
        |
        v
agent/data_agent.py
 - creates deterministic answer text
 - optionally asks Ollama for explanation, not calculation
        |
        v
pages/chat_page.py
 - renders answer, table, chart, chart insight, download, save, evaluation
```

## Who Does What?

| Work type | Responsible part |
| --- | --- |
| User interface | Streamlit pages and components |
| File upload and validation | `services/file_loader.py` |
| Dataset profiling | `services/dataset_profiler.py` |
| Data cleaning | `services/cleaning_service.py` |
| Natural-language planning | `agent/data_agent.py` and optionally Ollama |
| Safe tool selection | `agent/tool_registry.py` |
| Numeric calculation | `agent/tools.py` using Pandas/SciPy |
| Chart creation | `services/chart_service.py` using Plotly |
| Chart explanation | `services/chart_insight_service.py` |
| LLM explanation | `services/llm_service.py` with Ollama |
| Answer evaluation | `evaluation/` modules |
| Export/report generation | `services/export_service.py`, `services/report_service.py` |
| Session memory | `utils/session_state.py`, `agent/memory.py` |


## Slide: Technical Design and Responsibility Split

```text
User question
   |
   v
Streamlit chat UI
   |
   v
Data agent planner
   |
   |-- understands intent, columns, filters, aggregation, chart need
   |-- creates a validated AgentPlan
   v
Tool registry
   |
   |-- allows only approved tools
   |-- rejects unknown tools or invalid arguments
   v
Pandas / SciPy tools
   |
   |-- perform the actual calculation
   |-- return verified ToolResult data
   v
Plotly + insight service
   |
   |-- build chart from verified data
   |-- generate written explanation from chart evidence
   v
Reports and evaluation
```

Important message for the audience:

> The model does not calculate the final numbers. The model helps understand
> the user request and explain verified outputs. Pandas/SciPy performs the
> actual calculations.

## Model, LangChain, Ollama, Cleaning, and Evaluation

| Component | Role in this project |
| --- | --- |
| Ollama model | Runs the local LLM. It can help plan ambiguous natural-language questions and explain verified results. |
| LangChain / `langchain-ollama` | Provides the Python interface used to call the local Ollama chat model from `services/llm_service.py`. |
| LangGraph | Included in the stack as a possible agent-workflow framework, but the current submitted app mainly uses a custom deterministic orchestration flow in `agent/data_agent.py`. Do not claim LangGraph controls the runtime unless it is integrated later. |
| Agent planner | Converts user text into a safe `AgentPlan`: selected tool, arguments, optional chart spec, and response mode. |
| Tool registry | Acts like a security gate. Only allowlisted tools can run, and arguments are validated before execution. |
| Pandas/SciPy tools | Perform grouping, aggregation, filtering, statistics, correlations, outlier detection, and time-series calculations. |
| Plotly chart service | Converts verified tool results or selected dataset fields into visualizations. |
| Chart insight service | Reads chart evidence and writes key finding, supporting evidence, interpretation, caution, and next step. |
| Data cleaning service | Previews and applies reversible cleaning actions such as missing-value handling, duplicate removal, type conversion, and column fixes. |
| Evaluation service | Scores answers for correctness, relevance, faithfulness, completeness, tool accuracy, and chart accuracy using deterministic checks and optional LLM judging. |

## Technology Stack

- Python 3.11+
- Streamlit
- Pandas
- NumPy
- SciPy
- Plotly
- Pydantic
- LangChain, LangGraph, and `langchain-ollama`
- Ollama
- OpenPyXL and xlrd
- ReportLab
- pytest

## Setup

### macOS / Conda Setup

```bash
conda create -n data_agent python=3.11 -y
conda activate data_agent
pip install -r requirements.txt
cp .env.example .env

brew install ollama
ollama serve
ollama pull qwen2.5:1.5b

streamlit run app.py
```

`ollama serve` may not be required if the Ollama desktop application is already
running.

### Included Virtual Environment

```bash
source .venv/bin/activate
streamlit run app.py
```

To run on a specific port:

```bash
streamlit run app.py --server.port 8765
```

## Environment Configuration

The defaults in `.env.example` are:

```dotenv
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:1.5b
OLLAMA_FALLBACK_MODEL=llama3.2:1b
OLLAMA_TEMPERATURE=0.1
OLLAMA_NUM_CTX=4096
OLLAMA_TIMEOUT_SECONDS=120
MAX_UPLOAD_SIZE_MB=100
MAX_SAMPLE_ROWS_FOR_LLM=5
APP_ENV=development
```

## Supported Input Files

- `.csv`
- `.xlsx`
- `.xls`

Excel workbooks expose worksheet names. When the active worksheet changes, the
dataset profile and dataset-dependent chat context are refreshed.

## Main Features

- CSV, XLSX, and XLS upload
- worksheet selection for Excel files
- automatic schema profiling
- data-quality scoring
- missing-value analysis
- duplicate detection
- date, ID, categorical, and numeric column detection
- data-cleaning preview and confirmation
- automated EDA
- manual visualization builder
- natural-language chat analysis
- deterministic fallback when Ollama is offline
- Plotly chart generation
- chart insights
- safe Pandas operation preview
- saved insights
- answer evaluation
- report and export support
- benchmark test cases

## Example Questions

```text
Tell me about this dataset.
What are the data types of all columns?
Show total sales by region.
What percentage of total revenue comes from each region?
Which sub-category has the lowest profit?
Which ship mode generated the highest total sales?
Show total revenue and total profit over the years in one chart.
Show average unit price and average unit cost by item type.
Create a scatter plot of Sales versus Profit.
Create a box plot of Profit by Category.
Count distinct Order ID for each State.
How many unique orders were placed in 2017?
```

## Evaluation Button

The **Evaluate This Answer** button checks the quality of a chat answer. It is
mainly useful for debugging and improving the agent.

The evaluator considers:

- whether the selected tool was appropriate
- whether the answer matches the verified calculation
- whether the answer is relevant to the question
- whether the explanation is complete
- whether the chart matches the result
- whether errors and unsupported requests were handled clearly

The LLM may help judge answer quality when Ollama is online, but the actual
business calculation still comes from Pandas/SciPy tools.

## Safety Design

- no arbitrary code execution
- no `exec` or `eval`
- no unrestricted tool calls
- no full dataset sent to Ollama by default
- calculations are performed by approved Python functions
- tool arguments are validated before execution
- ambiguous requests ask for clarification instead of silently guessing
- uploaded dataset changes reset dataset-dependent context

## Testing

Run all tests:

```bash
pytest
```

Run the benchmark from the Evaluation page, or run it directly:

```bash
python -c "from evaluation.benchmark_runner import load_test_cases, run_benchmark_cases; print(run_benchmark_cases(load_test_cases('evaluation/test_cases.json'), 'sample_data'))"
```

## Current Limitations

- Natural-language parsing supports many common analytical patterns but not
  every possible wording.
- Ollama quality and latency depend on the local model and hardware.
- Inferred semantic types, dates, IDs, and outliers still require domain review.
- PNG export may require a compatible local Chrome installation.
- Multi-dataset joins are experimental and require explicit confirmation.
- Session memory is not persisted between Streamlit sessions.

## Future Improvements

- richer multi-filter query grammar
- more benchmark datasets with exact expected result tables
- stronger semantic matching for unseen business domains
- optional workspace persistence
- asynchronous model calls
- Docker deployment profile
- hosted Ollama-compatible endpoint support
