# Contenders see human-equivalent observations; privileged information is for training reward only

ViZDoom exposes exact coordinates, depth, object labels, sector data and save states, which would make the task far easier and the result far less credible. A Contender's inputs are limited to Human-equivalent Observations: screen pixels, HUD values, and the automap. Privileged Information may be used to compute reward during training and to draw debugging visualisations, because that mirrors a coach with a GPS training a runner who races without one. Every Scoreboard row states its observation class and every privileged reward term is declared in the milestone results.

## Consequences

- Early learning milestones may run a privileged Contender for teaching purposes, labelled as such on the Scoreboard.
- Go-Explore (M9) relies on save states; it is a training-time technique and the resulting policy is still evaluated without them.
