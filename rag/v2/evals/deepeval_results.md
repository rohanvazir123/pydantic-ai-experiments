# DeepEval RAG Evaluation Report

**Date:** 2026-06-29 04:29 UTC
**Corpus:** default:default
**RAG model:** `llama3.2:3b`
**Judge model:** `llama3.2:3b`
**Top-K retrieved:** 5
**Test cases:** 7

## Summary

| Metric | Avg Score | Threshold | Pass Rate | Status |
|--------|-----------|-----------|-----------|--------|
| Faithfulness | 0.801 | 0.7 | 71% (5/7) | ✅ Pass |
| Answer Relevancy | 0.762 | 0.7 | 71% (5/7) | ✅ Pass |
| Contextual Relevancy | 0.618 | 0.6 | 57% (4/7) | ✅ Pass |
| Contextual Precision | 0.500 | 0.6 | 50% (1/2) | ❌ Fail |
| Contextual Recall | 0.500 | 0.6 | 0% (0/2) | ❌ Fail |

## Per-Query Results

### Q1: What does NeuralFlow AI do?

**Tags:** `company`, `overview`
**Pipeline status:** `answered`  |  **Latency:** 11342 ms  |  **Confidence:** —

**Answer:**
> NeuralFlow AI is a cutting-edge AI automation agency that specializes in intelligent workflow automation, natural language processing solutions, and custom AI integration for enterprise clients. The company bridges the gap between artificial intelligence capabilities and real-world business needs by delivering solutions in various areas, including:

* Intelligent Document Processing: Automated extraction, classification, and analysis of business documents using advanced OCR and NLP techniques.
*…

**Expected (ground truth):**
> NeuralFlow AI builds AI-powered products and provides machine learning consulting services to enterprise clients.

| Metric | Score | Threshold | Pass |
|--------|-------|-----------|------|
| Faithfulness | 1.000 | 0.7 | ✅ |
| Answer Relevancy | 1.000 | 0.7 | ✅ |
| Contextual Relevancy | 0.571 | 0.6 | ❌ |
| Contextual Precision | 1.000 | 0.6 | ✅ |
| Contextual Recall | 0.500 | 0.6 | ❌ |

<details>
<summary>Judge reasoning</summary>

**Faithfulness:** The score is 1.00 because there are no contradictions found in the actual output.

**Answer Relevancy:** The score is 1.00 because there are no irrelevant statements in the actual output, making it a perfect answer that directly addresses the input question.

**Contextual Relevancy:** The score is 0.57 because the retrieval context contains many irrelevant statements, such as 'specializing in intelligent workflow automation' (which has no relation to NeuralFlow AI's focus on AI automation agency) and 'dedicate' (which is unrelated to the input), but only a few relevant statements like 'What does NeuralFlow AI do?' and 'NeuralFlow AI will maintain a competitive edge by investing in R&D' are present, indicating that the retrieval context has some relevance to the input.

**Contextual Precision:** The score is 1.00 because the relevant nodes are ranked higher than irrelevant nodes, with all 'yes' verdicts being ranked above the first 'no' verdict at node 5.

**Contextual Recall:** The score is 0.50 because the expected output lacks concrete details about NeuralFlow AI's product offerings and services, making it difficult to accurately attribute sentences to the node(s) in retrieval context.

</details>

**Retrieved context:** 5 chunk(s)
  - Chunk 1: NeuralFlow AI - Company Overview About Us NeuralFlow AI is a cutting-edge AI automation agency founded in 2023, speciali…
  - Chunk 2: NeuralFlow AI - Company Overview What We Do At NeuralFlow AI, we bridge the gap between artificial intelligence capabili…
  - Chunk 3: 1. System Overview The NeuralFlow AI platform is a comprehensive, cloud-native AI automation system designed for enterpr…
  - _(+2 more)_

---

### Q2: What is the PTO and leave policy?

**Tags:** `hr`, `benefits`
**Pipeline status:** `answered`  |  **Latency:** 5013 ms  |  **Confidence:** —

**Answer:**
> I can only answer questions about the knowledge base. Please ask about company policies, teams, documents, or business topics.

However, I can provide some information that might be relevant to the context, but it appears that this question is not related to any of the provided source passages. The provided source passages do not mention anything about PTO (paid time off) and leave policy explicitly.

**Expected (ground truth):**
> Employees receive paid time off that accrues over the year. The policy covers vacation days, sick leave, and public holidays.

| Metric | Score | Threshold | Pass |
|--------|-------|-----------|------|
| Faithfulness | 0.609 | 0.7 | ❌ |
| Answer Relevancy | 0.333 | 0.7 | ❌ |
| Contextual Relevancy | 0.214 | 0.6 | ❌ |
| Contextual Precision | 0.000 | 0.6 | ❌ |
| Contextual Recall | 0.500 | 0.6 | ❌ |

<details>
<summary>Judge reasoning</summary>

**Faithfulness:** The score is 0.61 because the actual output suggests that central banks aim to maintain a lean balance sheet, but this contradicts the intention to last for the duration of the crisis and be withdrawn once it's over.

**Answer Relevancy:** The score is 0.33 because there are no relevant statements in the actual output to increase the score, but the input question about PTO and leave policy is explicitly mentioned as not part of the knowledge base.

**Contextual Relevancy:** The score is 0.21 because most statements in the retrieval context are irrelevant, such as '46', 'credit losses', 'risk-taking', 'price stability', 'inflation', and 'public finances' (as quoted), which have no relation to PTO and leave policy, while only a few statements like 'The priority for monetary policy is to firmly re-establish price stability.' and 'Safety margins, or buffers, are essential for resilience.' are relevant to the input.

**Contextual Precision:** The score of 0.00 indicates that the relevant nodes are ranked lower than irrelevant nodes because the 'no' verdicts state that the context is unrelated to the topic of PTO and leave policy, specifically mentioning that the text discusses general considerations for policy without addressing PTO or leave policies (Rank 1), monetary policy without mentioning PTO or leave policies (Rank 2), crisis management without relevance to PTO and leave policies (Rank 3), near-term policy settings without mention of PTO or leave policies (Rank 4).

**Contextual Recall:** The score is 0.50 because most sentences in the expected output do not mention specific aspects of the retrieval context, such as balance sheet or key concepts, which are crucial for accurate contextual recall.

</details>

**Retrieved context:** 5 chunk(s)
  - Chunk 1: Central bank balance sheet choices Crisis management requires only temporarily larger balance sheets. Forceful balance s…
  - Chunk 2: Safety margins The post-GFC period highlights that there is a premium on retaining safety margins, ie  room  for  policy…
  - Chunk 3: Guiding considerations Any policy prescription depends on the balance of risks to the outlook. Although risks will  vary…
  - _(+2 more)_

---

### Q3: Which business units performed best in Q4?

**Tags:** `finance`, `q4`
**Pipeline status:** `answered`  |  **Latency:** 13088 ms  |  **Confidence:** —

**Answer:**
> Unfortunately, I can only answer questions about the knowledge base based on the provided source passages, and none of them provide direct information on which specific Business Units (BUs) within Tesla performed best in Q4 2024. However, I can offer some general context and insights.

The [q4-2024-business-review] document highlights the overall success of Q4 2024 but does not specify individual business unit performance. In contrast, the [meeting-notes-2025-01-08] document provides an overview…

| Metric | Score | Threshold | Pass |
|--------|-------|-----------|------|
| Faithfulness | 1.000 | 0.7 | ✅ |
| Answer Relevancy | 0.000 | 0.7 | ❌ |
| Contextual Relevancy | 0.789 | 0.6 | ✅ |

<details>
<summary>Judge reasoning</summary>

**Faithfulness:** The score is 1.00 because there are no contradictions found in the actual output.

**Answer Relevancy:** The score is 0.00 because there are no relevant statements in the actual output to assess, as the provided JSON does not contain any information that can be evaluated for relevance.

**Contextual Relevancy:** The score is 0.79 because the retrieval context contains relevant statements about commercial real estate risks and challenges in Q4, which are closely related to the input question about business units performing best in Q4, despite some irrelevant information (e.g., 'Challenges & Learnings', 'NON-GAAP FINANCIAL INFORMATION') being present.

</details>

**Retrieved context:** 5 chunk(s)
  - Chunk 1: Challenges & Learnings While Q4 was highly successful, we encountered several challenges that provide valuable insights:…
  - Chunk 2: Commercial real estate risks in the spotlight Declining  CRE  prices  have  increased  the  risk  of  default  and  loss…
  - Chunk 3: Q4 2024 Business Review Period: October 1 - December 31, 2024 NeuralFlow AI - Confidential…
  - _(+2 more)_

---

### Q4: What technologies and tools does the engineering team use?

**Tags:** `tech`, `engineering`
**Pipeline status:** `answered`  |  **Latency:** 8142 ms  |  **Confidence:** —

**Answer:**
> The NeuralFlow AI Engineering Team utilizes a variety of standard equipment to facilitate their work. According to the [NeuralFlow AI - Team Handbook], this includes:

* MacBook Pro or equivalent laptop (issued by the company, as part of an annual software license budget)
* External monitor (optional second monitor) for those who require it
* Keyboard and mouse of choice, allowing employees to tailor their input devices to their personal preferences
* Headphones for calls, providing essential ac…

| Metric | Score | Threshold | Pass |
|--------|-------|-----------|------|
| Faithfulness | 1.000 | 0.7 | ✅ |
| Answer Relevancy | 1.000 | 0.7 | ✅ |
| Contextual Relevancy | 0.846 | 0.6 | ✅ |

<details>
<summary>Judge reasoning</summary>

**Faithfulness:** The score is 1.00 because there are no contradictions found in the actual output.

**Answer Relevancy:** The score is 1.00 because there are no irrelevant statements in the actual output, making it a perfect answer that directly addresses the input question.

**Contextual Relevancy:** The score is 0.85 because the retrieval context contains relevant information about the engineering team's tools and technologies, such as '- MacBook Pro or equivalent laptop' and '- Annual software license budget ($500)', which are directly related to the input question.

</details>

**Retrieved context:** 5 chunk(s)
  - Chunk 1: NeuralFlow AI - Team Handbook Team Structure Engineering Team Led by our CTO, the engineering team comprises: - AI/ML En…
  - Chunk 2: NeuralFlow AI - Team Handbook Tools & Technology Standard Equipment All employees receive: - MacBook Pro or equivalent l…
  - Chunk 3: Technology Stack Enhancements Our engineering team implemented several critical infrastructure improvements to support s…
  - _(+2 more)_

---

### Q5: What are the onboarding steps for new employees?

**Tags:** `hr`, `onboarding`
**Pipeline status:** `answered`  |  **Latency:** 6271 ms  |  **Confidence:** —

**Answer:**
> **Onboarding Steps for New Employees**

The provided source documents outline the onboarding process for new employees in detail. 

During their first week, new hires will complete the following steps:

• **IT setup and equipment distribution**: The IT department is responsible for setting up the necessary systems, software, and equipment to support the new employee's work. This includes ensuring that they have all the required tools and devices to perform their job functions.
• 
The team assign…

| Metric | Score | Threshold | Pass |
|--------|-------|-----------|------|
| Faithfulness | 0.000 | 0.7 | ❌ |
| Answer Relevancy | 1.000 | 0.7 | ✅ |
| Contextual Relevancy | 0.463 | 0.6 | ❌ |

<details>
<summary>Judge reasoning</summary>

**Faithfulness:** The score is 0.00 because there are no contradictions found in the actual output to justify a higher faithfulness score.

**Answer Relevancy:** The score is 1.00 because there are no irrelevant statements in the actual output, making it a perfect answer that directly addresses the input question.

**Contextual Relevancy:** The score is 0.46 because most statements in the retrieval context are irrelevant to the input, with only a few statements like 'Your first week will include: - [ ] IT setup and equipment distribution' being somewhat relevant, but still not directly related to onboarding steps for new employees.

</details>

**Retrieved context:** 5 chunk(s)
  - Chunk 1: Employee Experience: - Staff redeployed from data entry to customer advisory roles - Employee sasfacon scores increased …
  - Chunk 2: NeuralFlow AI - Team Handbook Onboarding Checklist Your first week will include: - [ ] IT setup and equipment distributi…
  - Chunk 3: B. …but higher-income or better-educated workers expect greater benefits 2 1   See technical annex for details.     2   …
  - _(+2 more)_

---

### Q6: What are the company's goals and objectives for this year?

**Tags:** `strategy`, `goals`
**Pipeline status:** `answered`  |  **Latency:** 15266 ms  |  **Confidence:** —

**Answer:**
> The company's overall goals and objectives for this year (2025) can be summarized as follows:

**Company-Wide Objectives**

1. **Team Development**
	* Grow team from 47 to 75 employees
	* Achieve 90%+ employee satisfaction in quarterly surveys
	* Maintain engineering team diversity at 40%+ underrepresented groups
	* Promote 8+ team members to senior/leadership positions
	* Launch mentorship and leadership development programs, create clear career progression frameworks, expand learning budget an…

| Metric | Score | Threshold | Pass |
|--------|-------|-----------|------|
| Faithfulness | 1.000 | 0.7 | ✅ |
| Answer Relevancy | 1.000 | 0.7 | ✅ |
| Contextual Relevancy | 0.820 | 0.6 | ✅ |

<details>
<summary>Judge reasoning</summary>

**Faithfulness:** The score is 1.00 because there are no contradictions found in the actual output.

**Answer Relevancy:** The score is 1.00 because there are no irrelevant statements in the actual output, making it a perfect answer that directly addresses the input question.

**Contextual Relevancy:** The score is 0.82 because most of the retrieval context statements are highly relevant to the company's goals and objectives, such as 'Primary Objective: Achieve $12M in annual recurring revenue by end of 2025', '- Grow team from 47 to 75 employees', '- Achieve 90%+ employee satisfaction in quarterly surveys', and 'NeuralFlow AI - Mission and Goals' which contain specific targets, growth strategies, and company-wide initiatives that align with the input question.

</details>

**Retrieved context:** 5 chunk(s)
  - Chunk 1: NeuralFlow AI - Mission and Goals Company Goals 2024-2025 Team Development **Primary Objective** : Build world-class AI …
  - Chunk 2: NeuralFlow AI - Mission and Goals Company Goals 2024-2025 Revenue & Growth **Primary Objective** : Achieve $12M in annua…
  - Chunk 3: NeuralFlow AI - Mission and Goals Company Goals 2024-2025 Product Development **Primary Objective** : Launch three flags…
  - _(+2 more)_

---

### Q7: How does the performance review process work?

**Tags:** `hr`, `performance`
**Pipeline status:** `answered`  |  **Latency:** 11637 ms  |  **Confidence:** —

**Answer:**
> The performance review process at NeuralFlow AI is formal and occurs twice per year, in June and December. These reviews assess various aspects of an individual's performance, including:

* Technical competency and skill development
* Project delivery and impact
* Collaboration and communication
* Alignment with company values

These assessments aim to provide feedback that supports individual growth and helps refine the company's goals and objectives.

It appears that the formal review process …

| Metric | Score | Threshold | Pass |
|--------|-------|-----------|------|
| Faithfulness | 1.000 | 0.7 | ✅ |
| Answer Relevancy | 1.000 | 0.7 | ✅ |
| Contextual Relevancy | 0.625 | 0.6 | ✅ |

<details>
<summary>Judge reasoning</summary>

**Faithfulness:** The score is 1.00 because there are no contradictions found in the actual output.

**Answer Relevancy:** The score is 1.00 because there are no irrelevant statements in the actual output, indicating a clear and concise response to the input question about the performance review process.

**Contextual Relevancy:** The score is 0.62 because the retrieval context contains relevant statements about the performance review process (e.g., 'We conduct formal performance reviews twice per year...'), but these are overshadowed by irrelevant information (e.g., 'Reviews assess...' and 'community exploration').

</details>

**Retrieved context:** 5 chunk(s)
  - Chunk 1: NeuralFlow AI - Team Handbook Performance & Growth Review Cycle We conduct formal performance reviews twice per year (Ju…
  - Chunk 2: References Bender,  E  and  A  Koller  (2020):  'Climbing  towards  NLU:  on  meaning,  form  and understanding in the a…
  - Chunk 3: 1. Q4 2024 Product Performance Review Presented by: Jennifer Martinez & Kevin O'Brien…
  - _(+2 more)_

---

## Raw Scores (JSON)

```json
[
  {
    "query": "What does NeuralFlow AI do?",
    "tags": [
      "company",
      "overview"
    ],
    "status": "answered",
    "latency_ms": 11342,
    "scores": {
      "faithfulness": 1.0,
      "answer_relevancy": 1.0,
      "contextual_relevancy": 0.571,
      "contextual_precision": 1.0,
      "contextual_recall": 0.5
    }
  },
  {
    "query": "What is the PTO and leave policy?",
    "tags": [
      "hr",
      "benefits"
    ],
    "status": "answered",
    "latency_ms": 5013,
    "scores": {
      "faithfulness": 0.609,
      "answer_relevancy": 0.333,
      "contextual_relevancy": 0.214,
      "contextual_precision": 0,
      "contextual_recall": 0.5
    }
  },
  {
    "query": "Which business units performed best in Q4?",
    "tags": [
      "finance",
      "q4"
    ],
    "status": "answered",
    "latency_ms": 13088,
    "scores": {
      "faithfulness": 1.0,
      "answer_relevancy": 0.0,
      "contextual_relevancy": 0.789
    }
  },
  {
    "query": "What technologies and tools does the engineering team use?",
    "tags": [
      "tech",
      "engineering"
    ],
    "status": "answered",
    "latency_ms": 8142,
    "scores": {
      "faithfulness": 1.0,
      "answer_relevancy": 1.0,
      "contextual_relevancy": 0.846
    }
  },
  {
    "query": "What are the onboarding steps for new employees?",
    "tags": [
      "hr",
      "onboarding"
    ],
    "status": "answered",
    "latency_ms": 6271,
    "scores": {
      "faithfulness": 0.0,
      "answer_relevancy": 1.0,
      "contextual_relevancy": 0.463
    }
  },
  {
    "query": "What are the company's goals and objectives for this year?",
    "tags": [
      "strategy",
      "goals"
    ],
    "status": "answered",
    "latency_ms": 15266,
    "scores": {
      "faithfulness": 1.0,
      "answer_relevancy": 1.0,
      "contextual_relevancy": 0.82
    }
  },
  {
    "query": "How does the performance review process work?",
    "tags": [
      "hr",
      "performance"
    ],
    "status": "answered",
    "latency_ms": 11637,
    "scores": {
      "faithfulness": 1.0,
      "answer_relevancy": 1.0,
      "contextual_relevancy": 0.625
    }
  }
]
```
