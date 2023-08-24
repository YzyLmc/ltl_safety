# ltl_safety
Download [Virtual Home](https://github.com/xavierpuigf/virtualhome) simulator and dataset, and put them under `virtualhome/simulation` and `virtualhome/dataset` repectively.

### Changelog
08/02: `get_visible_nodes()` check indirect objects up to 3 levels.

08/05: These objects in TestScene1_graph mismatch the trimmed graph `{'wall_clock', 'food_cheese', 'towel_rack', 'bathroom_cabinet', 'after_shave', 'kitchen_counter', 'home_office', 'dining_room', 'food_food', 'coffe_maker', 'filing_cabinet', 'bathroom_counter', 'measuring_cup', 'food_carrot'}`. Among them, two rooms `kitchen` and `living_room` have been renamed to `dining_room` and `home_office`.

08/07: `max_nodes` set to 500 instead of 300.

08/08: 
- Switching back to v2.3.0. Constructing new dataset from 1.0.0 dataset. Aiming at 50 programs * 10 constraint for each program.
- Action embeddings stored on Oscar (ada-002)

08/09: deleting dining_room and home_office in `resources/class_name_equivalence.json`. Change class_name `livingroom` to `living_room` in `virtualhome_v2.3.0/env_graphs/TestScene1_graph.json`

08/10: More programs added for env0. Program `0_13.txt` need all lightswitch to be turned off at the beginning. Test run with `src/exp_virtualhome.py`, output not guranteed to be runable. 

08/11: delay_react and prompt_react doesn't work since `stop` doesn't need to check DFA and cannot check using the current method, i.e., if there's a path towards accepting states.

08/12: potential issues moving forward: providing env info to base agent. and Initial state. Instant reaction has issue, e.g., go to kitchen right after visiting living room is impossible if the kitchen between them is tracked too.

08/23: change 'U' to 'W' for past avoidance.