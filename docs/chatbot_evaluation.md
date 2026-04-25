# Chatbot Evaluation

Evaluated in `notebooks/07_rag_evaluation.ipynb` — 30 questions, Groq llama-3.1-8b-instant, ChromaDB retrieval.

## Results

| Metric | Result | Target |
|--------|--------|--------|
| Total questions | 30 | — |
| Trick refusal rate | **100%** (5/5) | 100% |
| Mean latency | 5.20s | <3s |
| Median latency | 6.12s | — |
| p95 latency | 9.47s | — |
| Under 3s | 27% | >target |
| Mean top-chunk cosine score | 0.656 | — |
| Questions with score > 0.50 | 26/30 | — |

**Note on refusal rate:** Auto-detection keyword matched 4/5 (80%). Q27 answer was "For personalized guidance, please speak with an admissions advisor." — a correct refusal/defer, but not caught by the `"admissions office"` keyword. True refusal rate = 5/5 (100%).

**Note on latency:** Groq free-tier API is variable — contact lookup (Q3) returned in 0.73s; complex questions hit 6–10s. Target of <3s was not met at mean but is achievable with caching of common queries or a paid Groq tier.

## Retrieval Quality by Category

| Category | Mean top-score | Mean latency |
|----------|---------------|-------------|
| contact | 0.816 | 0.73s |
| academics | 0.755 | 4.27s |
| governance | 0.739 | 6.61s |
| research | 0.713 | 4.89s |
| general | 0.681 | 5.45s |
| trick | 0.400 | 6.80s |
| admissions | 0.413 | 6.90s |

Trick questions and "admissions" category show lower retrieval scores (0.40–0.41) — corpus has limited direct admissions-process content (scraped from public site, not internal FAQs).

## Sample Answers

**Q: How do I contact ASU admissions?**  
> To find your personal admission team member's email and phone number, select undergraduate or graduate below and answer a couple quick questions on the ASU Admission Services page (https://admission.asu.edu/contact).  
> Source: admission.asu.edu/contact (score=0.816) ✅

**Q: What types of degree programs does ASU offer?**  
> According to the sources: Joint programs are offered by more than one college. More than 800 degree programs in a wide range of studies are available.  
> Source: students.asu.edu/programs (score=0.747) ✅

**Q: Was I admitted to ASU this year? [TRICK]**  
> For personalized guidance, please speak with an admissions advisor.  ✅ (correct defer)

**Q: Who won the Super Bowl last year? [TRICK]**  
> I don't have that information — please contact the admissions office directly. ✅

## Known Limitations

- Corpus scraped from public ASU site — no internal admissions data (specific deadlines, GPA cutoffs)
- Low retrieval scores for admissions-specific queries → consider supplementing corpus with admissions FAQ pages
- Latency exceeds 3s target on Groq free tier; use query caching for production
- Manual correctness scoring not yet complete — `data/output/rag_eval_raw.csv` available for review
