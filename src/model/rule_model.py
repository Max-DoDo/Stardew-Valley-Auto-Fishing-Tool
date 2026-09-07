

from src.model.state import State


class RuleModel:

    def action(self, state: State) -> int:
        if not state.isFishing:
            return 0

        dead_zone = 6
        # fish_y 更小，说明鱼在绿色条上方
        # 星露谷里按住会让绿色条上升
        if state.distance < -dead_zone:
            return 1

        # fish_y 更大，说明鱼在绿色条下方
        # 松开会让绿色条下降
        if state.distance > dead_zone:
            return 0

        # 距离很近时，先松开，后面可以改成保持上一动作
        return 0
    
    def reset_episode(self):
        pass
