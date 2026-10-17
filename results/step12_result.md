# Step 12: Test the cell-type label on mixed-type data (Bugeon et al. 2022)
Date: 2026-10-17

## What was run
- Task 0 only (access check). Following the step's rule ("if the network blocks it, STOP"), no data was downloaded and no analysis was run
- The figshare dataset (https://doi.org/10.6084/m9.figshare.19448531) is not reachable from this environment. Every request was refused by the session's egress proxy:

  | URL tried | Result |
  |---|---|
  | `https://api.figshare.com/v2/articles/19448531` (file list API) | `curl: (56) CONNECT tunnel failed, response 403` (proxy: `connect_rejected`, host not allowed) |
  | `https://doi.org/10.6084/m9.figshare.19448531` (DOI from the task) | `CONNECT tunnel failed, response 403` |
  | `https://figshare.com/articles/dataset/19448531` (landing page) | `CONNECT tunnel failed, response 403` |
  | `https://ndownloader.figshare.com/articles/19448531/versions/1` (bulk download) | `CONNECT tunnel failed, response 403` |

- No other routes (mirrors, third-party copies) were tried, as instructed
- The NeuPRINT / NuCLR / NetFormer code repositories were not cloned. Their loaders would only show the expected format, not give the data, and the data block makes every later task impossible

## Data
- Experiment id: Bugeon et al. 2022, figshare article 19448531. Not accessed
- Cre line / area / depth: not accessed. From the task description: mouse V1, one GCaMP for all neurons; subclasses Lamp5, Pvalb, Vip, Sncg, Sst plus pyramidal; identified afterwards by in situ transcriptomics
- Neurons: not accessed. From the task description: ~89 identified inhibitory and ~393 pyramidal cells per session, 1,090 unique inhibitory cells, 4 mice, 17 sessions
- Frames / frame rate / duration: unknown. The file list (names, sizes, URLs) could not be obtained, because the figshare API itself is blocked

## Results
- None. Tasks 1–3 (preparation, true-label ridge / MLP test, inferred labels) were not started, and no numbers are reported
- No mock or substitute data was used

**What is needed to continue:**
- Allow these hosts in the cloud environment's network settings: environment menu in the session title bar → Edit → Network access, then either a broader access level or these hosts under Allowed domains. Steps: https://code.claude.com/docs/en/cloud-environments#network-access
  - `api.figshare.com`: file list and metadata
  - `figshare.com` and `ndownloader.figshare.com`: file downloads. Figshare downloads may also redirect to an S3 storage host (e.g. `s3-eu-west-1.amazonaws.com`); this could not be checked because the first request is already refused
  - `doi.org`: optional, only to resolve the DOI
- Or put the needed files (activity traces, cell IDs / transcriptomic labels, stimulus and behaviour info, timing) into `data/bugeon/` by another route, and the step can start from there
- Optional, to read the format: `github.com` is reachable from this session (this repository is pushed through the same proxy), so the NeuPRINT, NuCLR or NetFormer data loaders could be read once their repository URLs are confirmed

## Plots
- None (no data)

## Problems / errors
- Network policy of this environment denies all figshare hosts (HTTP 403 on CONNECT through the agent proxy). This is a policy block, not a transient error: the same proxy reaches PyPI and GitHub, and it reached the Allen S3 mirror in steps 10–11

## What I noticed
- The mixed-type test cannot be run here until figshare is allowed. Steps 1–11 (Allen, single-line recordings) stay the only evidence so far. The question of whether the cell-type label helps when types share one recording and one indicator is still open.
