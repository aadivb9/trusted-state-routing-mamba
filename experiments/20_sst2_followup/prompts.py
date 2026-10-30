"""Three predefined formats; current-example labels never enter rendering."""
FORMATS=('published_answer_cue','explicit_one_word','four_shot')
LABELS={0:'negative',1:'positive'}

def render_parts(data,format_id,text):
 instruction=data['instruction']
 if format_id=='published_answer_cue':return instruction+'\nText: ',text,'\nSentiment:'
 if format_id=='explicit_one_word':return instruction+'\nRespond with one word: positive or negative.\nReview: ',text,'\nAnswer:'
 if format_id=='four_shot':
  prefix='Classify each movie review as positive or negative.\n'
  for demo in data['demonstrations']:prefix+='Review: '+demo['text']+'\nSentiment: '+LABELS[demo['label']]+'\n\n'
  return prefix+'Review: ',text,'\nSentiment:'
 raise ValueError('Unknown frozen prompt format')
