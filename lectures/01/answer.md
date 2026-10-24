# Lecture 01 Answers

## Exercises

- Model's answer to the course-specific question (the 3 homework levels; a summary is fine):
- What you noticed when comparing it with the correct meaning in the glossary:
- Token counts you displayed (input / output):

## Homework

- Model used: gemini-3.5-flash-lite
- Level attempted: Basic
- What you did: I asked the same question, “What is LoRA?”, three times.
- Results: The responses conveyed the same core concepts but differed slightly in their style of explanation.
- Observations and questions: Each request used the same number of input tokens but generated a different number of output tokens.

- Level attempted: Implementation and Advanced
- What you did: With the help of an AI coding assistant, I modified the code to retain conversation history and measure the response time for each question.
- Results: The number of input tokens increased with each follow-up question because previous questions and answers were included in subsequent requests to the model. This increased token usage may cause token-based rate limits on the free tier to be reached sooner. However, response times varied and did not increase consistently as the conversation grew longer.
- Observations and questions: These results suggest that factors other than input length may also affect response time.