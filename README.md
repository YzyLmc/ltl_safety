# ltl_safety
Download [Virtual Home](https://github.com/xavierpuigf/virtualhome) simulator and dataset, and put them under `virtualhome/simulation` and `virtualhome/dataset` repectively.

### Changelog
08/02: `get_visible_nodes()` check indirect objects up to 3 levels.

08/05: These objects in TestScene1_graph mismatch the trimmed graph `{'wall_clock', 'food_cheese', 'towel_rack', 'bathroom_cabinet', 'after_shave', 'kitchen_counter', 'home_office', 'dining_room', 'food_food', 'coffe_maker', 'filing_cabinet', 'bathroom_counter', 'measuring_cup', 'food_carrot'}`. Among them, two rooms `kitchen` and `living_room` have been renamed to `dining_room` and `home_office`.

08/07: `max_nodes` set to 500 instead of 300.