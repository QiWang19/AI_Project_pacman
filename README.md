# AI_Project_pacman

## Minimax:

iteration process

(adversrial search) pacman->max agent, ghost-> min agent

tree structure: one layer max agent, several min agent(number of ghosts)

## alpha-beta prun:

detail process link:https://www.youtube.com/watch?v=xBXHtz4Gbdo&t=351s

compare and update the alpha, beta

alpha-max value

beta-min value

## Expectimax:

adversary is not that worse as min agent. Use weght to calculate the value of expect agent.

max agent got the max of them.

can not prun

## value-iteration:

## MDP: Bellman equation

## Q-Learning

Don't know T, R; Different update q value from MDP
How to explore: freedy
Keep a table of all a-values

## Approximate Q-Learning: Generalize

can not explore every state; Compute q value using features and their weights;
update the weights;

