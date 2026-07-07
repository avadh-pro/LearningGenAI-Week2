# Basic Prompt Engineering Techniques

Prompt engineering is like learning the secret handshake to get the best out of large language models like OpenAI's GPT series. It's the art of sweet-talking an AI into giving you exactly what you want—without it going on a philosophical rant about the meaning of life. Let's dive into this mystical art with some fun techniques and a Python demo to make you feel like an AI whisperer.

## What is Prompt Engineering?

Imagine you're at a restaurant, and the waiter is a super-intelligent AI. If you just say, "Food, please," you're probably going to end up with a surprise dish. Prompt engineering is like crafting a detailed order: "I'd like a Ghee Podi Idli, please." With great prompts come great results (and fewer surprises).

## Key Techniques in Prompt Engineering

### 1. Zero-shot Prompting

This technique is the YOLO (You Only Live Once) of AI interactions: no examples, just vibes. You simply ask a question and hope the AI gets it right. Perfect for when you're feeling lucky.

### 2. Few-shot Prompting

Few-shot prompting is like showing the AI a couple of magic tricks and then asking it to perform the next one. By providing a few examples, you're saying, "Here's how it's done—now your turn."

### 3. Prompt Chaining

Prompt chaining is the AI equivalent of a treasure hunt. You give it one clue (prompt), then use its answer to move to the next clue. It's a step-by-step process, and by the end, you've struck informational gold.

### 4. Chain-of-Thought (CoT) Reasoning

CoT reasoning is like asking the AI to show its work, math-class style. Instead of jumping to conclusions, it walks you through the logic step by step. This technique can make even the trickiest problems feel like a stroll in the park.

## Python Demo: Implementing Prompt Engineering Techniques

Enough talk—let's get our hands dirty (or at least, our keyboards). Here's a Python script that showcases these techniques using OpenAI's API (you can use Cohere or any other open-source model from Hugging Face as well). For the notebook link, click here.

### Prerequisites

- Install the OpenAI Python library:

```bash
pip install openai
```

- Obtain an OpenAI API key from OpenAI's platform.

### The Code

#### 1. Zero Shot Prompting

```python
def zero_shot_prompt(prompt):
    """Demonstrate zero-shot prompting."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print("Zero-shot Prompting:\n", response.choices[0].message.content)


zero_shot_prompt("Explain the concept of quantum computing in simple terms.")
```

```
## Response
Zero-shot Prompting:
Quantum computing is a type of computing that uses the principles of quantum mechanics to process information in a fundamentally different way than traditional computers. Here are some key points to understand:

1. **Bits vs. Qubits**: Traditional computers use bits as the smallest unit of information, which can be either a 0 or a 1. Quantum computers use qubits (quantum bits), which can be both 0 and 1 at the same time due to a property called superposition. This allows quantum computers to process a large amount of information simultaneously.

2. **Entanglement**: Qubits can be entangled, meaning that the state of one qubit can depend on the state of another, no matter how far apart they are. This connection can lead to faster processing and more complex calculations that wouldn't be possible with traditional bits.

3. **Parallelism**: Because qubits can represent multiple states at once, quantum computers can perform many calculations at the same time. This gives them the potential to solve certain problems much faster than classical computers.

4. **Quantum Gates**: Just like classical computers use logical gates to process bits, quantum computers use quantum gates to manipulate qubits. These gates change the probabilities associated with each qubit's state, allowing for complex computations.

In summary, quantum computing harnesses the strange and powerful rules of quantum mechanics to perform calculations in a way that classical computers cannot, potentially leading to breakthroughs in fields like cryptography, chemistry, and optimization problems.
```

#### 2. One Shot Prompting

```python
def few_shot_prompt(prompt):
    """Demonstrate few-shot prompting."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print("Few-shot Prompting:\n", response.choices[0].message.content)


prompt = ("Translate the following English phrases to French:\n"
          "English: Hello, how are you?\n"
          "French: Bonjour, comment ça va?\n"
          "English: What is your name?\n"
          "French: Comment vous appelez-vous?\n"
          "English: Where is the library?\n"
          "French:")
few_shot_prompt(prompt)
```

```
## Response
Few-shot Prompting:
French: Où est la bibliothèque?
```

#### 3. Prompt Chaining

```python
def chain_part1(initial_prompt):
    """Demonstrate prompt chaining."""
    # First prompt
    response1 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": initial_prompt}]
    )
    languages = response1.choices[0].message.content
    return languages


def chain_part2(follow_up_prompt):
    # Second prompt using the output of the first
    response2 = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": follow_up_prompt}]
    )
    return response2.choices[0].message.content


initial_prompt = "Provide a list of three popular programming languages."
response_part1 = chain_part1(initial_prompt)
follow_up_prompt = f"Explain the primary use case for each of these programming languages:\n{response_part1}"
response_part2 = chain_part2(follow_up_prompt)
print("Part 1:\n", response_part1, "\n")
print("-" * 150, "\n")
print("Part 2:\n", response_part2)
```

```
## Response
Part 1:
Three popular programming languages are:
1. Python
2. JavaScript
3. Java

------------------------------------------------------------------------------------------------------------------------------------------------------

Part 2:
Certainly! Here's a brief overview of the primary use cases for each of the three programming languages:

### 1. Python
**Primary Use Case: Data Science and Machine Learning**
- Python is widely used in data analysis, data visualization, machine learning, and artificial intelligence. With libraries such as NumPy, Pandas, TensorFlow, and Scikit-learn, it provides robust tools for statistical analysis, data manipulation, and building machine learning models. Additionally, Python's simple syntax and readability make it an excellent choice for both beginners and experienced developers working on data-centric projects.

### 2. JavaScript
**Primary Use Case: Web Development**
- JavaScript is primarily used for front-end web development, enabling interactive and dynamic content on websites. It runs in the browser and allows developers to create engaging user interfaces. With frameworks and libraries like React, Angular, and Vue.js, JavaScript also plays a crucial role in building single-page applications (SPAs). Additionally, with the advent of Node.js, JavaScript can also be used for server-side programming, making it a full-stack development language.

### 3. Java
**Primary Use Case: Enterprise Applications**
- Java is commonly used in building large-scale enterprise applications due to its stability, scalability, and portability. It runs on the Java Virtual Machine (JVM), which allows applications to run on any platform that supports Java. It is often used for backend systems, Android app development, and large distributed systems, thanks to its robustness and the extensive ecosystem of frameworks like Spring and Hibernate that facilitate enterprise-level development.

These primary use cases highlight the strengths of each language in different domains.
```

#### 4. Chain of Thought

```python
def chain_of_thought(prompt):
    """Demonstrate chain-of-thought reasoning."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    print("Chain of Thought:\n", response.choices[0].message.content)


prompt = ("A train leaves the station at 60 miles per hour. Another train leaves the same station "
          "one hour later at 80 miles per hour. How long will it take for the second train to catch the first train? "
          "Explain your reasoning step by step.")
chain_of_thought(prompt)
```

```
## Response
Chain of Thought:
To solve this problem, we'll set up the situation step by step.

1. **Define Variables**:
   - Let \(t\) be the time in hours that the first train has been traveling when the second train catches up to it.
   - The first train travels at 60 miles per hour.
   - The second train travels at 80 miles per hour and leaves the station one hour after the first train.

2. **Determine the Distance of the First Train**:
   - The first train leaves the station and travels for \(t\) hours.
   - The distance it travels is given by the formula:
     \[
     \text{Distance} = \text{Speed} \times \text{Time} = 60t \text{ miles}
     \]

3. **Determine the Distance of the Second Train**:
   - The second train leaves the station one hour after the first train. So, it will have been traveling for \((t - 1)\) hours when it catches up with the first train.
   - The distance traveled by the second train is:
     \[
     \text{Distance} = \text{Speed} \times \text{Time} = 80(t - 1) \text{ miles}
     \]

4. **Set Up the Equation**:
   - For the second train to catch the first train, the distances they travel must be equal:
     \[
     60t = 80(t - 1)
     \]

5. **Solve the Equation**:
   \[
   60t = 80t - 80
   \]
   - Rearranging the equation:
     \[
     80t - 60t = 80
     \]
     \[
     20t = 80
     \]
     \[
     t = \frac{80}{20} = 4
     \]
```
