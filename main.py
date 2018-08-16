import textplayer.textPlayer as tP
import argparse
import agents.vivan_agent as vivan_agent
import sys
#from keras import backend as K
#from keras.layers import Dense, Flatten
#from keras.models import Sequential
#from keras_helper import NNWeightHelper
from snes import SNES
import pickle
#import gc
from timeit import default_timer as timer
#from guppy import hpys
#import pdb

## below for debugging
#sys.argv += 'zork1.z5'
game_chosen = 'detective.z5'

def save(generations=None,track_scores=None):
	if generations in [99,149,199,249,299,349,399]:
		# save the scores
		with open('track_scores_detective_pop10gen100_trial_160818.pkl', 'wb') as f:
			pickle.dump(track_scores, f)

#def dump_garbage():
#	"""
#	show us what's the garbage about
#	"""
#		
#	# force collection
#	print "\nGARBAGE:"
#	gc.collect()
#
#	print "\nGARBAGE OBJECTS:"
#	for x in gc.garbage:
#		s = str(x)
#		if len(s) > 80: s = s[:80]
#		print type(x),"\n  ", s
		
#def manual_play(game_chosen):
#	textPlayer = tP.TextPlayer(game_chosen)
#	# pass the variable to another class
#	state = textPlayer.run()
#	print (state)
#	action = raw_input('>> ')
#	last_score = 0

	# we specify that exit is a way to quit game, so highlight obviously here
#	while action!= 'exit':
#		state = textPlayer.execute_command(action)
#		print(state)
#		if textPlayer.get_score() != None:
#			score, __ = textPlayer.get_score()
#			reward = score - last_score
#			last_score = score
#			print  ('Your overall score is {0} and you gained reward of {1} in the last action'.format(score,reward))
#		action = raw_input('>> ')

#	textPlayer.quit()

#def train_agent(game_chosen):


#hp = hpy()  # type: object
#before = hp.heap()

# start desired game file
textPlayer = tP.TextPlayer(game_chosen)

# defining the NN structure for neurovolui
#model = Sequential()
#model.add(Dense(100,input_shape=(100,)))
#model.add(Dense(100))
#model.compile(loss="mse", optimizer="adam")
#nnw = NNWeightHelper(model)
#weights = nnw.get_weights()

#hp.setrelheap()
track_scores = []
last_score = 0
# defining population size
pop_size = 10
# defining generation size
generations = 100
# init the agent
agent = vivan_agent.vivAgent()
initial_weights = agent.agent_return_weights()

state = textPlayer.run()
last_state = state
# print (state)
# pass variables to SNES
snes = SNES(initial_weights, 1, pop_size)

start = timer()
#while (counter < training_cycles):
for i in range(0, generations):
	asked = snes.ask()
	told = []
	j = 0
	for asked_j in asked:
		# use SNES to set the weights of the NN
		# only run the SNES after 1st round
		# pass asked to function in vivan_wordfiles.py everytime
		# init the neural network in vivan_wordfiles.py
		# but it is run only when the def_results_for_words is called.
		# how about reward?

		# set the last state as previous state of last round
		last_state = state
		action = agent.take_action(state,asked_j,False)
		state = textPlayer.execute_command(action)
		print('::: {0} >>> {1}'.format(action,state))
		print('This is Population No. {0} of Generation no. {1}'.format(j,i))
		#try:
		if textPlayer.get_score() != None:
			#try:
			score, possible_score = textPlayer.get_score()
			# if previous state is equal to current state, then agent reward is -0.2 otherwise reward * 1.2
			reward = score - last_score
			if last_state == state:
				agent_reward = reward - 0.2
			else:
				agent_reward = reward * 1.2
			told.append(agent_reward)
			last_score = score
			agent.update(reward)
			accumulated_reward = agent.get_total_points_earned()
			print  ('Your overall score is {0} and you gained reward of {1} in the last action and agent reward of {2}'.format(accumulated_reward,reward,agent_reward))
			track_scores.append((state,j,i,action,agent_reward,reward,accumulated_reward))
			# except:
			# 	score = 0
			# 	# if previous state is equal to current state, then agent reward is -0.2 otherwise reward * 1.2
			# 	reward = score - last_score
			# 	if last_state == state:
			# 		agent_reward = reward - 0.2
			# 	else:
			# 		agent_reward = reward * 1.2
			# 	told.append(agent_reward)
			# 	last_score = score
			# 	agent.update(reward)
			# 	accumulated_reward = agent.get_total_points_earned()
			# 	print  ('Your overall score is {0} and you gained reward of {1} in the last action and agent reward of {2}'.format(accumulated_reward,reward,agent_reward))
			# 	track_scores.append((state,j,i,action,agent_reward,reward,accumulated_reward))
		else:
			score = 0
			# if previous state is equal to current state, then agent reward is -0.2 otherwise reward * 1.2
			reward = score - last_score
			if last_state == state:
				agent_reward = reward - 0.2
			else:
				agent_reward = reward * 1.2
			told.append(agent_reward)
			last_score = score
			agent.update(reward)
			accumulated_reward = agent.get_total_points_earned()
			print  ('Your overall score is {0} and you gained reward of {1} in the last action and agent reward of {2}'.format(accumulated_reward,reward,agent_reward))
			track_scores.append((state,j,i,action,agent_reward,reward,accumulated_reward))
		j += 1
	snes.tell(asked,told)
	save(i,track_scores)
# set the weights of the NN
snes_centre_weights = snes.center
# pass this weights to the nnw file
agent.pass_snes_centre_weight(snes_centre_weights)
#nnw.set_weights(snes.center)
#save(counter,track_scores)
final_weights = agent.agent_return_weights()
word_seen = agent.agent_return_word_seen()
agent.agent_return_models() 

# save the scores
with open('track_scores_detective_pop10gen100_trial_160818.pkl', 'wb') as f:
	pickle.dump(track_scores, f)

# save the final weights
with open('final_weights_detective_pop10gen100_trial_160818.pkl', 'wb') as g:
	pickle.dump(final_weights, g)

# save the words seen in the game
with open('words_seen_detective_pop10gen100_trial_160818.pkl', 'wb') as h:
	pickle.dump(word_seen, h)

# save the models seen in the game
#with open('trained_model_detective_pop10gen100_trial_160818.pkl', 'wb') as i:
#	pickle.dump(trained_model, i)

end = timer()

print(end - start)

#h = hp.heap()
#pdb.set_trace()


#textPlayer.quit()


# for debugging
#if __name__ == '__main__':
#	manual_play(game_chosen)
	#gc.enable()
	#gc.set_debug(gc.DEBUG_LEAK)
#	train_agent(game_chosen)

	# show the dirt ;-)
	#dump_garbage()
	#train_agent(game_chosen)

#parser = argparse.ArgumentParser(description='Which game do you want to play?')
#parser.add_argument('file',help='Specify game name?')
#function_map = {'manual_play' : manual_play,
#                'train_agent' : train_agent }

#parser.add_argument('command', choices=function_map.keys(),help = "Type manual_play if you want to manually play the game or " 
#					" train_agent to train the agent")

# parse arguments
#args = parser.parse_args()
#game_chosen = args.file
#func = function_map[args.command]
#func(game_chosen)
# future: add if neuroevolution is on, specify training cycles, population, generation, plot tsne?  