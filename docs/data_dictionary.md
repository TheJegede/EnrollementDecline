# Data Dictionary

Documents every field across the four data streams used by the project. All sources are public-domain or generated synthetically — no proprietary or PII data is involved.

## Source registry

| Source | Path | License | Acquisition |
|---|---|---|---|
| NCES Digest tables 219.10, 302.10, 303.10 | `data/raw/nces/` | Public domain (US Government work) | `src.data_acquisition.fetch_nces_tables()` |
| IPEDS surveys ADM, IC, EFFY (FY 2018–2022) | `data/raw/ipeds/` | Public domain (US Government work) | `src.data_acquisition.fetch_ipeds()` |
| Synthetic applicant set | `data/synthetic/applicants.csv` | Generated locally; reproducible with `random_state=42` | `src.data_synthesis.generate_applicants()` |
| ASU admissions corpus | `data/corpus/` | Public scrape, robots.txt respected, throttled 1 req/s. Institution name redacted in deployed dashboard. | `src.data_acquisition.scrape_asu_corpus()` |

## NCES Digest tables

Each `.xls` file is a multi-row-header pivot. Use `pd.read_excel(..., header=None)` and slice headers manually.

| File | Title | Key columns (logical) |
|---|---|---|
| `tabn219.10.xls` | High school graduates, by state | year, state, public/private, count |
| `tabn302.10.xls` | Recent high school completers and their enrollment in college, by sex and race/ethnicity | year, sex, race, completer count, enrolled count |
| `tabn303.10.xls` | Total fall enrollment in degree-granting postsecondary institutions, by attendance status, sex, and age | year, attendance status, sex, age band, count |

Phase 2 forecasting parses these into a tidy `data/processed/` long-form CSV (year, segment, value).

## IPEDS surveys

Each zip contains a CSV plus a data dictionary HTML/PDF. Year suffix corresponds to the survey reference year.

### ADM — Admissions

| Column (illustrative) | Description |
|---|---|
| `UNITID` | IPEDS institution ID |
| `APPLCN` | Total applications |
| `APPLCNM`, `APPLCNW` | Applications by sex |
| `ADMSSN` | Total admits |
| `ENRLT` | Total enrolled |
| `SATVR25`, `SATVR75` | SAT verbal 25th / 75th percentile |
| `ACTCM25`, `ACTCM75` | ACT composite percentiles |

Use to compute institution-level yield = `ENRLT / ADMSSN`.

### IC — Institutional Characteristics

| Column (illustrative) | Description |
|---|---|
| `UNITID` | IPEDS institution ID |
| `INSTNM` | Institution name |
| `CONTROL` | 1=public, 2=private nonprofit, 3=private for-profit |
| `ICLEVEL` | 1=4-year, 2=2-year, 3=less-than-2-year |
| `OBEREG` | OBE region (geographic) |

Used to enrich ADM rows with institution type for segment-level forecasting.

### EFFY — 12-month enrollment by race/ethnicity

| Column (illustrative) | Description |
|---|---|
| `UNITID` | IPEDS institution ID |
| `EFFYALEV` | Aggregation level |
| `EFFYLEV` | Level of student (undergrad / grad) |
| `EFYTOTLT` | Total enrollment |
| `EFYAIANT`, `EFYASIAT`, `EFYBKAAT`, `EFYHISPT`, `EFYNHPIT`, `EFYWHITT`, `EFY2MORT`, `EFYUNKNT`, `EFYNRALT` | Counts by race/ethnicity (American Indian, Asian, Black, Hispanic, Native Hawaiian/PI, White, Two-or-more, Unknown, Nonresident) |

## Synthetic applicants — `data/synthetic/applicants.csv`

50,000 inquiry-level rows. Each row is a prospective lead at one of five institution segments. Three sequential outcomes are nested: `inquired_to_applied → admitted → admit_to_enroll`.

### Identity

| Column | Type | Description |
|---|---|---|
| `applicant_id` | int | Sequential 0..49999 |
| `institution_segment` | str | One of `R1`, `regional_state`, `private_lac`, `community_college`, `online` |

### Demographics (bias-audit relevant)

| Column | Type | Values |
|---|---|---|
| `race_ethnicity` | str | White / Hispanic / Black / Asian / Other |
| `gender` | str | F / M |
| `first_gen_flag` | int (0/1) | First-generation college student |
| `income_band` | str | low / middle / high |
| `region` | str | Northeast / Midwest / South / West |

### Academics

| Column | Type | Description |
|---|---|---|
| `hs_gpa` | float | 0.0–4.0, Beta-distributed; mean per segment |
| `sat_score` | int | 400–1600, correlated with GPA |

### Geography

| Column | Type | Description |
|---|---|---|
| `distance_miles` | float | 1–3000, lognormal |

### Engagement

| Column | Type | Description |
|---|---|---|
| `campus_visit_flag` | int (0/1) | Visited campus before applying |
| `email_engagement_score` | float | 0–100 |
| `financial_aid_inquiry_flag` | int (0/1) | Inquired about aid |
| `days_since_first_inquiry` | int | 1–365 |

### Application context

| Column | Type | Description |
|---|---|---|
| `intended_major` | str | Business / Engineering / CS / Nursing / Education / Liberal Arts / Sciences / Undecided |
| `application_date_relative_to_deadline` | int | Negative = before deadline; mean −21 |
| `source_channel` | str | organic_search / paid_ad / fair / referral / highschool_visit / email_campaign |

### Yield-only features (zero unless `admitted == 1`)

| Column | Type | Description |
|---|---|---|
| `aid_package_amount` | float | USD; depends on segment + income |
| `scholarship_offer_flag` | int (0/1) | |
| `days_to_deposit_deadline` | int | 1–90 |
| `peer_admit_count` | int | Poisson(3) — how many peers also admitted |

### Targets

| Column | Type | Description |
|---|---|---|
| `inquired_to_applied` | int (0/1) | Lead model target — overall ~30% positive |
| `admitted` | int (0/1) | Conditional on `inquired_to_applied == 1`; per-segment quantile |
| `admit_to_enroll` | int (0/1) | Yield model target — calibrated per segment to ±2pp of IPEDS averages |

### Calibration targets

| Segment | Target yield | Tolerance |
|---|---|---|
| R1 | 33% | ±2pp |
| regional_state | 22% | ±2pp |
| private_lac | 28% | ±2pp |
| community_college | 50% | ±2pp |
| online | 40% | ±2pp |

Calibration is performed by bisecting the logit intercept until the empirical positive rate matches the target. Random seed: `42`.

## ASU corpus — `data/corpus/`

| Field | Description |
|---|---|
| Filename | URL-encoded source URL (truncated to 200 chars) + `.md` |
| First line | `# Source: <url>` — original URL preserved for citation |
| Body | Plain markdown extracted via trafilatura (links stripped) |

Allowed domains: `admission.asu.edu`, `students.asu.edu`, `tuition.asu.edu`, `asu.edu`. Crawl is BFS one-hop deep from `ASU_SEED_URLS`. Robots.txt enforced via custom Disallow parser. 1 req/sec throttle. Default cap: 250 pages, files <200 chars dropped.

Phase 4 chunks each markdown file into ~500-token passages with 50-token overlap, embeds with `all-MiniLM-L6-v2`, and persists to ChromaDB at `data/vector_db/`.
