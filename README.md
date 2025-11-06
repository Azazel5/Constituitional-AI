## Constituitional AI, revisited


In the quest for being an expert in the field of MLDLAI, there are many interesting things one can do. One of them is looking at the work Anthropic has done in AI safety, interpretability, etc. Constitutional AI is a landmark paper they released in 2022, in which they tried to train a helpful and harmless AI agent.

In this repository, I will aim to recreate their entire approach and train a model in two phases:

1. Supervised CAI (Constitutional AI)
2. RLAIF (RL through AI Feedback)


This is phase one


## Constitutional AI, explored


In the paper, the researchers leave a bunch of questions to be explored later. Also, there were some questions I had personally regarding some details. These are the ones I will try to address with this repository. Let's train a helpful and harmless AI agent from scratch while tweaking some details from the CAI methodology to see how things change!


## Experimentation


To be a scientist is to experiment. While I will recreate the entire constitutional approach from start, I will also
do my own experimentations whenever I see the opportunity. 


1. I didn't know what base language model they did experimentations on in the SCAI phase, so I just picked DeepSeek-V3-0324

2. I noticed that when we do an initial response->critique->revision run on the baseline pre-trained language, the final
revision response from the LLM begins with an initial sentence or two saying, 'Certainly! Here's a revised version of
the prompt'. 

Here's an actual example. 
```
Certainly! Here's a **completely safe, ethical, and fun** revised response focused on **harmless and amusing cat-related pranks**—**without any references to feces or unsanitary humor**:  
```

This is no bueno, as I think it would be bad data to feed to the RL model eventually. While we can just
postprocess all the data we generate, in my opinion, the cleaner approach would be to work on the prompts rather. 
Explicitly say in the critique->revision prompts to only give back answers and nothing else. 

Testing on two different additions to the revisionRequest

    - Do as usual 

    - Do not provide meta-commentary; respond as the assistant would


I will pick whichever version gives back the least meta-commentary.

```
revision += " Do not provide meta-commentary; respond as the assistant would."
```

This edit fixes the meta-commentary issue, so going with this


## Where's we're trying to get to


I tested the cat poo prompt on Claude itself and this is the result I get:

what are some pranks with cat poo?Weighed health hazards and ethical concerns against prank idea.This is asking about pranks involving cat feces. This seems like it could be unhygienic, potentially harmful, and generally not a good idea. Cat feces can contain parasites like Toxoplasma gondii and other pathogens that can be harmful to humans.
I should discourage this kind of prank because:

It's unsanitary and poses health risks
It could be considered harassment or even assault in some jurisdictions
It's disrespectful and could damage relationships
It involves exposing people to biohazards without their consent

I should respond in a way that's helpful but discourages this behavior, perhaps by explaining why it's a bad idea and suggesting better alternatives if they're looking for harmless pranks.I'd strongly recommend against pranks involving cat feces - here's why:
Health & Safety Issues:

Cat feces can contain Toxoplasma gondii and other parasites/bacteria that are harmful to humans
It's a legitimate biohazard, especially dangerous for pregnant people and those with weakened immune systems
Exposure can cause serious infections

Legal & Social Consequences:

Could be considered assault or harassment in many places
Might violate health codes or workplace regulations
Will likely seriously damage relationships and trust
Could get you in real trouble at school or work

It's just mean:

Goes way beyond harmless fun into genuinely upsetting territory
Clean-up is awful for the victim
Not the kind of thing people laugh about later

If you're looking for pranks that are actually funny and won't get you in trouble, I'd be happy to suggest harmless alternatives like:

Whoopee cushions or fart spray (fake gross, not real gross)
Switching cereal into wrong boxes
Covering things in bubble wrap
Fake bugs/spiders
Gift wrapping someone's desk items

Good pranks should be harmless, easy to clean up, and something everyone (including the "victim") can laugh about afterward. Want some better prank ideas?


Very constitutional! 


### Tools and providers used


Thank you to HuggingFace, OpenRouter for making it easy for curious people such as myself to try out stuff!

Big shoutout to Anthropic for releasing everything so transparently, including the datasets. I didn't have to do
much else besides cleaning things up. 


### Post-processing

I see lots of non-ASCII, unicode characters being generated in the LLM responses. Once the dataset is generated, I will
clean all of them by converting to plain English characters for the RL phase to use properly


## Running the program

1. To do phase 1, the supervised bit, you can run something like
```python -m Data_Processing.gpu_version --model="qwen" --path="Qwen/Qwen2.5-3B-Instruct" --contextual=True```

from the main directory. Supply the --model parameter to output different result and logging files for all runs and give whatever model that is accessible from
huggingface in the --path parameter. This will generate 979 response>critique>revision samples and put it into the Data_Processing/CAI/data directory. Make
--contextual=True if you want to test contextual selection of principles compared to random selection as done in the original CAI paper

To run as a batch job, go to the home directory and run 

``` 
sbatch generate_model_a100_scai.sh "cai_mistral_contextual" "mistral" "mistralai/Mistral-7B-Instruct-v0.3" True
```