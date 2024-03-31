from langchain.prompts import FewShotChatMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, SystemMessagePromptTemplate
from opentrons import protocol_api
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAI
from langchain_community.llms import HuggingFaceEndpoint
from dataclasses import dataclass
import os, textwrap, subprocess, re
from pathlib import Path
from langchain_openai import ChatOpenAI

@dataclass
class Templates():
  system:str=""
  few_shot:str=""
  code:str=""
  input:str=""

  @classmethod
  def collect_templates(cls, agent:str) -> dataclass:
    cls.system=cls.get_template(agent, 'system')
    cls.direction=cls.get_template(agent, 'direction')
    cls.code=cls.get_template(agent, 'code')
    return cls

  def get_template(agent:str, type:str) -> str:
    '''Helper function for downloading template types
    Parameters:
    ----------
    agent: ['action' or 'critic']
    type: ['code', 'direction', 'system']

    Outputs:
    --------
    LLM prompts as strings
    '''
    filepath=Path(f"/nfs/lambda_stor_01/homes/bhsu/michigan_demo/{agent}_{type}.txt")
    with open(filepath, 'r') as file:
      prompt = file.read()
    return prompt
  

class LabWorker():
    def __init__(self, temperature=0, model_type=None, verbose=True):
        self.temperature = temperature
        self.model_type = model_type
        self.verbose=verbose
        if self.model_type=='gemini': 
            self.llm = GoogleGenerativeAI(model="gemini-pro",
                                        google_api_key=api_key, 
                                        verbose=self.verbose)
        elif self.model_type=='GPT': 
            self.llm = ChatOpenAI(model='gpt-4', 
                                  api_key=api_key, 
                                  verbose=self.verbose)

    @classmethod
    def initialize(cls, temperature:int=0, model_type:str='gemini',
                   verbose:bool=True, templates=None) -> object:
        """Creates an instance of the agent and initializes a chains"""
        agent = cls(temperature, model_type, verbose)
        agent.init_chain(templates)
        return agent

    # def generate(self, input:str) -> str:
    #     return self.chain.run({'user_input':input})

    def init_chain(self, templates:dataclass) -> None:
        """Initialized LLMChain based on agent_type [action, critic]"""
        system_prompt = SystemMessagePromptTemplate.from_template(templates.system)
        few_shot_prompt=(
            HumanMessagePromptTemplate.from_template(templates.direction)
            + AIMessagePromptTemplate.from_template(templates.code))
        history_prompt = MessagesPlaceholder(variable_name="chat_history")
        memory_gen = ConversationBufferWindowMemory(k=10, memory_key = "chat_history", return_messages = True)
        input_prompt = HumanMessagePromptTemplate.from_template("{user_input}")
        self.final_prompt = (
                      system_prompt
                      + few_shot_prompt
                      + history_prompt
                      + input_prompt
                      )
        self.chain = LLMChain(llm = self.llm,
                              prompt = self.final_prompt,
                              verbose = self.verbose,
                              memory = memory_gen)
        return None



class ActionAgent(LabWorker):
    def candidate_script_extractor(self, raw_edits: str) -> str:
      """Takes in raw re-edit output from GPT and returns cleaned script
      Parameters
      =========
      raw_edits: str
          str that contains raw code with comments output from GPT
      Output
      ========
      format_code: str
          str that contains formatted code that can be ran in cli
      """
      pattern = re.escape("```") + r"(.*?)" + re.escape("```")
      matches = re.search(pattern, raw_edits, flags=re.DOTALL)
      if matches is None:
          return raw_edits
      format_code = matches.group(1).strip() 
      return format_code

    def simulate_opentrons(self, code: str, storage_path:str) -> str:
      """Helper function that simulates code in opentrons
      Parameters
      ==========
      code: str
          code generatred from code_generator chain
          as a str
      save_path: str
          save path as a str
      Output
      =========
      str:
          str for output error or output
      """
      code = textwrap.dedent(code)
      with open(storage_path, "w") as file:
          file.write(code)
      cli = ["opentrons_simulate", storage_path]
      result = subprocess.run(args=cli, capture_output=True, text=True)
      return result

    def refine(self, code: str, num_retries:int, storage_path:str) -> str:
      """Function that takes in code and iteratively refines it by saving as a file and running 
      it against the cli for execution errors. 
      Parameters
      ==========
      code: str
        code as a string 
      num_retries: int 
        number times to refine code 
      storage_path: str
        file path to store temporary code 
      Output
      ======
      tuple[str]: 
        stores code and final message"""
      
      code = self.candidate_script_extractor(code)
      result = self.simulate_opentrons(code, storage_path)
      for i in range(num_retries):
          if result.returncode==0:
              break
          error_message = """The script you provided failed, and the execution error is described below. Please fix the error.
          \n\nError Message:\n\n""" + result.stderr
          raw_edit = self.chain.run({'user_input': error_message})
          code = self.candidate_script_extractor(raw_edit)
          result = self.simulate_opentrons(code, storage_path)
      success=result.returncode==0
      if success and i == 0:
          message = f"Your code did not need to be fixed:\n{result.stdout}"
      elif success:
          message = f"Your code fixed itself in {i} tries:\n{result.stdout}"
      else:
          message = f"Your code failed to fix itself in {i} times:\n{result.stderr}"
        #   raise ValueError("Candidate script failed after {num_retries} attempts. Script: {cand_script_path}")
      return code, message
    

if __name__=="__main__": 

    my_model = 'Gemini'
    api_key = None

    if my_model=='Gemini':
        llm = GoogleGenerativeAI(model="gemini-pro",
                            google_api_key=api_key)
    elif my_model=="GPT": 
        llm = ChatOpenAI(model="gpt-4", openai_api_key=api_key)

    action_template = Templates.collect_templates('action')
    action_template.input='''
    1.) Load a themocycler module of type 'thermocyclerModuleV2'.
    2.) Loop through the following set of sub-tasks in a loop 20 times:
        a.) Set the thermocycler block temperature to 95 C, set hold time to 30 seconds, and set block max volume to 32 ul.
        b.) Set the thermocycler block temperature to 57 C, set hold time to 30 seconds, and set block max volume to 32 ul.
        c.) Set the thermocycler block temperature to 72 C, set hold time to 60 seconds, and set block max volume to 50 ul.
    '''

    actionagent = ActionAgent.initialize(templates=action_template, model_type='gemini')

    code = actionagent.chain.run({'user_input':action_template.input}) 

    code, result = actionagent.refine(code, 7, "/nfs/lambda_stor_01/homes/bhsu/michigan_demo/something.py") 

    critic_template = Templates.collect_templates('critic')
    criticagent = LabWorker.initialize(templates=critic_template, model_type='GPT')
    critique = criticagent.chain.run({'user_input': action_template.input +'\n' + result[0]})

    print('Done')




