# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# # Create a Graph from a text

# This notebook demonstrates how to extract graph from any text using the graph maker
#
# Steps:
# - Define an Ontology
# - Load a list of example text chunks. We will use the Lord of the Rings summary from this wikipedia page. 
# - Create Graph using an Open source model using Groq APIs. 
# - Save the graph to Neo4j db
# - Visualise
#
#
#
# Loading the graph maker functions ->

from knowledge_graph_maker import GraphMaker, Ontology, GroqClient, OpenAIClient
from knowledge_graph_maker import Document

# # Define the Ontology. 
#
# The ontology is a pydantic model with the following schema. 
#
# ```python
# class Ontology(BaseModel):
#     label: List[Union[str, Dict]]
#     relationships: List[str]
# ```
#
#

# For example lets use summaries of the LOTR books from the Wikipedia page. I have copied it into a file for easy import

from lotr_wikipedia_summary import lord_of_the_rings_wikipedia_summary as example_text_list
len(example_text_list)

# Here is the ontology we will use for the LOTR summaries ->

# +

ontology = Ontology(
    labels=[
        {"Person": "Person name without any adjectives, Remember a person may be referenced by their name or using a pronoun"},
        {"Object": "Do not add the definite article 'the' in the object name"},
        {"Event": "Event event involving multiple people. Do not include qualifiers or verbs like gives, leaves, works etc."},
        "Place",
        "Document",
        "Organisation",
        "Action",
        {"Miscellaneous": "Any important concept can not be categorised with any other given label"},
    ],
    relationships=[
        "Relation between any pair of Entities"
        ],
)

# -

# ## Select a Model
#
# Groq support the following models at present. 
#
# *LLaMA3 8b*
# Model ID: llama3-8b-8192
#
# *LLaMA3 70b*
# Model ID: llama3-70b-8192
#
# *Mixtral 8x7b*
# Model ID: mixtral-8x7b-32768
#
# *Gemma 7b*
# Model ID: gemma-7b-it
#
#
# Selecting a model for this example ->
#

# +

## Groq models
model = "mixtral-8x7b-32768"
# model ="llama3-8b-8192"
# model = "llama3-70b-8192"
# model="gemma-7b-it"

## Open AI models
oai_model="gpt-3.5-turbo"

## Use Groq
# llm = GroqClient(model=model, temperature=0.1, top_p=0.5)
## OR Use OpenAI
llm = OpenAIClient(model=oai_model, temperature=0.1, top_p=0.5)

# -

# ## Create documents out of text chumks. 
# Documents is a pydantic model with the following schema 
#
# ```python
# class Document(BaseModel):
#     text: str
#     metadata: dict
# ```
#
# The metadata we add to the document here is copied to every relation that is extracted out of the document. More often than not, the node pairs have multiple relation with each other. The metadata helps add more context to these relations
#
# In this example I am generating a summary of the text chunk, and the timestamp of the run, to be used as metadata. 
#

# +
import datetime
current_time = str(datetime.datetime.now())


graph_maker = GraphMaker(ontology=ontology, llm_client=llm, verbose=False)

def generate_summary(text):
    SYS_PROMPT = (
        "Succintly summarise the text provided by the user. "
        "Respond only with the summary and no other comments"
    )
    try:
        summary = llm.generate(user_message=text, system_message=SYS_PROMPT)
    except:
        summary = ""
    finally:
        return summary


docs = map(
    lambda t: Document(text=t, metadata={"summary": generate_summary(t), 'generated_at': current_time}),
    example_text_list
)

# -

# ## Create Graph
# Finally run the Graph Maker to generate graph. 

# +

graph = graph_maker.from_documents(
    list(docs), 
    delay_s_between=0 ## delay_s_between because otherwise groq api maxes out pretty fast. 
    ) 
print("Total number of Edges", len(graph))
# -

for edge in graph:
    print(edge.model_dump(exclude=['metadata']), "\n\n")

# # Save the Graph to Neo4j 

# +
from knowledge_graph_maker import Neo4jGraphModel

create_indices = False
neo4j_graph = Neo4jGraphModel(edges=graph, create_indices=create_indices)

neo4j_graph.save()

# -


