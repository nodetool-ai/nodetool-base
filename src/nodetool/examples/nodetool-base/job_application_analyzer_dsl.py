"""
Job Application Analyzer DSL Example

Analyze job descriptions and provide personalized application advice.

Workflow:
1. **Job Description Input** - Provide job posting
2. **Requirements Analysis** - Extract key requirements
3. **Skills Matching** - Compare with candidate skills
4. **Gap Analysis** - Identify skill gaps
5. **Application Tips** - Generate tailored advice
6. **Resume Suggestions** - Recommend resume highlights
"""

from nodetool.dsl.graph import create_graph, run_graph
from nodetool.dsl.nodetool.input import StringInput
from nodetool.dsl.nodetool.agents import Agent, Classifier
from nodetool.dsl.nodetool.text import FormatText
from nodetool.dsl.nodetool.output import StringOutput
from nodetool.metadata.types import LanguageModel, Provider


# Job description
job_posting = StringInput(
    name="job",
    description="Job description to analyze",
    value="""
Position: Senior Data Scientist

Company: TechCorp Inc.

Requirements:
- 5+ years experience in data science and machine learning
- Strong Python programming skills
- Experience with TensorFlow and PyTorch
- SQL and database management expertise
- Experience with cloud platforms (AWS, GCP, or Azure)
- Excellent communication and presentation skills
- Background in statistics and linear algebra

Responsibilities:
- Develop machine learning models for predictive analytics
- Lead data science team
- Present findings to stakeholders
- Collaborate with engineering teams

Nice to have:
- PhD in related field
- Experience with big data technologies
- Published research papers
""",
)

# Candidate profile
candidate_profile = StringInput(
    name="profile",
    description="Candidate skills and background",
    value="""
Skills:
- 4 years as Data Analyst (not full Data Scientist)
- Proficient in Python and SQL
- Some TensorFlow experience
- AWS experience (EC2, S3)
- Strong statistics background

Education:
- Master's in Statistics
- Not a PhD

Experience:
- Worked at 2 startups
- Led small analytics projects
- Limited team leadership experience
""",
)

# Analyze job requirements
requirements_analyzer = Agent(
    prompt=FormatText(
        template="""Analyze this job posting and extract:
1. Core technical skills required
2. Experience level needed
3. Soft skills emphasized
4. Nice-to-have qualifications
5. Company culture indicators

Job:
{{ job }}""",
        job=job_posting.output,
    ).output,
    model=LanguageModel(
        type="language_model",
        id="gpt-4o-mini",
        provider=Provider.OpenAI,
    ),
    system="You are an HR analyst specializing in job descriptions.",
    max_tokens=600,
)

# Assess candidate fit
fit_assessor = Agent(
    prompt=FormatText(
        template="""Compare this candidate profile to the job requirements and provide:
1. Alignment score (0-100%)
2. Strengths matching the role
3. Skill gaps to address
4. How to present experience favorably
5. Interview preparation tips

Candidate:
{{ profile }}

Job Requirements (Summary):
- 5+ years as Data Scientist
- Advanced Python/ML skills
- TensorFlow and PyTorch
- Cloud platform experience
- Team leadership experience""",
        profile=candidate_profile.output,
    ).output,
    model=LanguageModel(
        type="language_model",
        id="gpt-4o",
        provider=Provider.OpenAI,
    ),
    system="You are a career coach. Provide honest, actionable advice for job applicants.",
    max_tokens=800,
)

# Generate application strategy
application_advice = FormatText(
    template="""# Job Application Analysis Report

## Position: Senior Data Scientist at TechCorp Inc.

## Job Requirements Summary:
{{ requirements }}

## Candidate Assessment:
{{ fit_assessment }}

## Recommended Application Strategy:

### Resume Optimization:
- Highlight Python and statistics expertise prominently
- Emphasize AWS and machine learning projects
- Frame data analyst role as preparation for senior role
- Include any projects involving TensorFlow

### Cover Letter Focus:
- Acknowledge experience gap but show growth trajectory
- Emphasize learning ability and passion for ML
- Highlight successful project outcomes
- Show understanding of company's needs

### Interview Preparation:
1. Deep dive on TensorFlow and PyTorch projects
2. Prepare to discuss transition from analyst to scientist role
3. Research company's current data challenges
4. Prepare questions about team structure and growth opportunities

### Skill Gap Closure:
- Consider online certification in PyTorch
- Brush up on big data technologies (Spark, Hadoop)
- Practice presenting technical findings
- Prepare portfolio project using required tech stack

## Overall Recommendation:
**Moderate-to-Good Fit** - You have strong fundamentals but need to address experience gap.
Focus on demonstrating learning ability and relevant project experience.

## Success Probability: 60-70% with proper preparation
""",
    requirements=requirements_analyzer.out.text,
    fit_assessment=fit_assessor.out.text,
)

# Output the analysis
output = StringOutput(
    name="application_analysis",
    value=application_advice.output,
)

# Create the graph
graph = create_graph(output)


if __name__ == "__main__":
    result = run_graph(graph)
    print("Job Application Analysis:")
    print(result)
