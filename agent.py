import re
from dotenv import load_dotenv
from openai import OpenAI
from tools import search

load_dotenv()
client = OpenAI()


ACTION_PATTERN = re.compile(r'Action:\s*Search\["(.+?)"\]', re.IGNORECASE | re.DOTALL)


class Agent:
    def __init__(self, system_prompt, model="gpt-4o-mini", max_iterations=5):
        self.system = system_prompt
        self.model = model
        self.max_iterations = max_iterations
        self.messages = [{"role": "system", "content": self.system}]

    def construct_prompt(self, query):
        self.messages.append({"role": "user", "content": query})
        return self.messages

    def _call_llm(self, stop=None):
        return client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            stop=stop,
        )

    def _extract_action_query(self, output):
        match = ACTION_PATTERN.search(output)
        if not match:
            return None
        return match.group(1).strip()

    def execute(self, query):
        self.construct_prompt(query)

        last_output = ""
        iteration = 0
        while iteration < 5:
            response = self._call_llm(stop=["Observation:", "\nObservation:"])

            output = (response.choices[0].message.content or "").strip()
            output = output.replace("Action: Final Answer:", "Final Answer:")
            last_output = output
            print(output)
            self.messages.append({"role": "assistant", "content": output})

            if "Final Answer:" in output:
                return output

            action_query = self._extract_action_query(output)
            if not action_query:
                reminder = (
                    "Observation: No valid Action detected. "
                    "Please respond with either Action: Search[\"...\"] or Final Answer:."
                )
                print(reminder)
                self.messages.append({"role": "user", "content": reminder})
                iteration += 1
                continue

            result = search(action_query)
            observation_text = f"Observation: {result}"
            print(observation_text)
            self.messages.append({"role": "user", "content": observation_text})

            if result.startswith("Error:") or "No results found" in result:
                reflection_prompt = (
                    "Observation quality is low. Reflect and try a new query with different keywords. "
                    "Do not repeat the same Action."
                )
                print(reflection_prompt)
                self.messages.append({"role": "user", "content": reflection_prompt})

            iteration += 1

        synthesize_prompt = (
            "You reached max iterations. Provide Final Answer using gathered observations only. "
            "If evidence is insufficient or conflicting, say so explicitly."
        )
        self.messages.append({"role": "user", "content": synthesize_prompt})
        final_response = self._call_llm()
        final_output = (final_response.choices[0].message.content or "").strip()
        final_output = final_output.replace("Action: Final Answer:", "Final Answer:")
        print(final_output)
        self.messages.append({"role": "assistant", "content": final_output})
        return final_output or last_output or "Final Answer: Unable to determine with confidence."