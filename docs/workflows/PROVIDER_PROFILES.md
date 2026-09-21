# Provider profiles

The runtime supports two named DashScope workspaces without storing API keys in
the repository:

| Profile | API-key environment variable | Base-URL environment variable |
| --- | --- | --- |
| `myx` (default) | `DASHSCOPE_API_KEY` | `DASHSCOPE_BASE_URL` |
| `lwj` | `LWJ_DASHSCOPE_API_KEY` | `LWJ_DASHSCOPE_BASE_URL` (required) |

The historical `myx` default remains backward-compatible. New provider
endpoints and all API keys must remain in environment variables and must not be
committed.

For commands built with `add_config_arguments`, select the workspace explicitly:

```powershell
$env:LWJ_DASHSCOPE_API_KEY = "<secret>"
$env:LWJ_DASHSCOPE_BASE_URL = "<workspace OpenAI-compatible URL>"
python -m multi_dataset_diverse_rl.cli --provider_profile lwj <other arguments>
```

Use `--provider_profile myx` (or omit the option) for the existing workspace.
Role-specific `--solver_api_key_env`, `--optimizer_api_key_env`, and evaluator
counterparts remain available as explicit overrides.

Older standalone scripts that use the shared credential resolver can be
switched process-wide:

```powershell
$env:DASHSCOPE_PROVIDER_PROFILE = "lwj"
$env:LWJ_DASHSCOPE_API_KEY = "<secret>"
$env:LWJ_DASHSCOPE_BASE_URL = "<workspace OpenAI-compatible URL>"
python scripts/<runner>.py <arguments>
```

The selected profile name and endpoint identity enter run identity and provider
accounting. Secret values do not.
