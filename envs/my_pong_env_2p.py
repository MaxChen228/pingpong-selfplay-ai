import gym
from gym import spaces
import numpy as np
import pygame
import random
import math

from envs.physics import collide_sphere_with_moving_plane

class PongEnv2P(gym.Env):
    """
    雙人 Pong：
      - 上板(A) / 下板(B)
      - 觀測 7 維: (ball_x, ball_y, ball_vx, ball_vy, my_paddle_x, other_paddle_x, spin)
      - 每彈 bounce_count++；若 bounce_count % speed_scale_every==0 => vx,vy *= (1+speed_increment)
      - 馬格努斯效應 + 剛體碰撞
    """

    def __init__(self,
                 render_size=400,
                 paddle_width=0.2,
                 paddle_speed=0.02,
                 max_score=3,
                 enable_render=False,

                 enable_spin=True,
                 magnus_factor=0.01,
                 restitution=0.9,
                 friction=0.2,
                 ball_mass=1.0,
                 world_ball_radius=0.03,

                 ball_speed_range=(0.01, 0.05),
                 spin_range=(-10, 10),
                 ball_angle_intervals=None,

                 speed_scale_every=3,
                 speed_increment=0.2
                 ):
        super().__init__()
        self.render_size = render_size
        self.paddle_width = paddle_width
        self.paddle_speed = paddle_speed
        self.max_score = max_score
        self.enable_render = enable_render

        # 馬格努斯 & 剛體參數
        self.enable_spin = enable_spin
        self.magnus_factor = magnus_factor
        self.restitution = restitution
        self.friction = friction
        self.ball_mass = ball_mass
        self.world_ball_radius = world_ball_radius

        # 初始速度/自轉
        self.ball_speed_range = ball_speed_range
        self.spin_range = spin_range
        self.ball_angle_intervals = ball_angle_intervals if ball_angle_intervals else [[-60, -30],[30,60]]

        # 每彈幾次就加速
        self.speed_scale_every = speed_scale_every
        self.speed_increment = speed_increment
        self.bounce_count = 0  # 記錄球撞板次數

        # 動作空間： (A, B) 各 {0=左,1=不動,2=右}
        self.action_space = spaces.MultiDiscrete([3,3])

        # 觀測空間 => 7 維
        # (ball_x, ball_y, ball_vx, ball_vy, my_paddle_x, other_paddle_x, spin)
        # x,y ∈ [0,1], vx,vy ∈ [-1,1], paddle_x ∈ [0,1], spin ∈ [-10,10]
        low  = np.array([0,   0,   -1, -1, 0,   0,   -10], dtype=np.float32)
        high = np.array([1,   1,    1,  1, 1,   1,    10], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

        if self.enable_render:
            pygame.init()
            self.screen = pygame.display.set_mode((self.render_size, self.render_size))
            pygame.display.set_caption("Pong 2P - 7D Obs + SpeedScale + Spin")
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.scoreA = 0
        self.scoreB = 0
        self.bounce_count = 0  # reset bounce 計數

        # 擋板位置
        self.top_paddle_x = 0.5
        self.bottom_paddle_x = 0.5

        # 球位置 => 中間
        self.ball_x = 0.5
        self.ball_y = 0.5

        # 隨機速度
        speed = random.uniform(*self.ball_speed_range)

        # 隨機角度 => 兩個區間
        if random.random()<0.5:
            angle_deg = random.uniform(*self.ball_angle_intervals[0])
        else:
            angle_deg = random.uniform(*self.ball_angle_intervals[1])
        angle_rad = math.radians(angle_deg)

        self.ball_vx = speed * math.cos(angle_rad)
        self.ball_vy = speed * math.sin(angle_rad)

        # spin
        self.spin = random.uniform(*self.spin_range)
        self.spin_angle = 0.0  # 用於繪製十字

        return self._get_obs()

    def step(self, actionA, actionB):
        # 更新擋板
        if actionA == 0:
            self.top_paddle_x -= self.paddle_speed
        elif actionA == 2:
            self.top_paddle_x += self.paddle_speed
        self.top_paddle_x = np.clip(self.top_paddle_x, 0, 1)

        if actionB == 0:
            self.bottom_paddle_x -= self.paddle_speed
        elif actionB == 2:
            self.bottom_paddle_x += self.paddle_speed
        self.bottom_paddle_x = np.clip(self.bottom_paddle_x, 0, 1)

        rewardA = 0.0
        rewardB = 0.0
        done = False

        # 馬格努斯
        if self.enable_spin:
            self.ball_vx += self.magnus_factor * self.spin * self.ball_vy

        # 更新球
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # 左右牆
        if self.ball_x < 0:
            self.ball_x = -self.ball_x
            self.ball_vx *= -1
        elif self.ball_x > 1:
            self.ball_x = 2 - self.ball_x
            self.ball_vx *= -1

        # 檢查上板
        if self.ball_y < 0:
            paddle_min = self.top_paddle_x - self.paddle_width/2
            paddle_max = self.top_paddle_x + self.paddle_width/2
            if paddle_min <= self.ball_x <= paddle_max:
                # 撞到上板 => 剛體碰撞
                u = 0.0
                if actionA == 0: u = -self.paddle_speed
                elif actionA == 2: u = self.paddle_speed

                vn = self.ball_vy
                vt = self.ball_vx
                omega = self.spin

                vn_post, vt_post, omega_post = collide_sphere_with_moving_plane(
                    vn, vt, u, omega,
                    self.restitution,
                    self.friction,
                    self.ball_mass,
                    self.world_ball_radius
                )
                self.ball_vy = vn_post
                self.ball_vx = vt_post
                self.spin = omega_post
                self.ball_y = 0.0

                # 計數 => 若 bounce_count 達 speed_scale_every => vx, vy 加速
                self.bounce_count += 1
                self._maybe_scale_speed()
            else:
                # 沒接到 => B +1
                rewardA -= 1.0
                rewardB += 1.0
                self.scoreB += 1
                if self.scoreB >= self.max_score:
                    done = True
                return self._get_obs(), (rewardA, rewardB), done, {}

        # 檢查下板
        elif self.ball_y > 1:
            paddle_min = self.bottom_paddle_x - self.paddle_width/2
            paddle_max = self.bottom_paddle_x + self.paddle_width/2
            if paddle_min <= self.ball_x <= paddle_max:
                # 撞到底板 => 剛體碰撞
                u = 0.0
                if actionB == 0: u = -self.paddle_speed
                elif actionB == 2: u = self.paddle_speed

                vn = -self.ball_vy
                vt = self.ball_vx
                omega = self.spin

                vn_post, vt_post, omega_post = collide_sphere_with_moving_plane(
                    vn, vt, u, omega,
                    self.restitution,
                    self.friction,
                    self.ball_mass,
                    self.world_ball_radius
                )
                self.ball_vy = -vn_post
                self.ball_vx = vt_post
                self.spin = omega_post
                self.ball_y = 1.0

                self.bounce_count += 1
                self._maybe_scale_speed()
            else:
                # B沒接 => A +1
                rewardA += 1.0
                rewardB -= 1.0
                self.scoreA += 1
                if self.scoreA >= self.max_score:
                    done = True
                return self._get_obs(), (rewardA, rewardB), done, {}

        return self._get_obs(), (rewardA, rewardB), done, {}

    def _maybe_scale_speed(self):
        """檢查 bounce_count 是否達到整除 speed_scale_every => vx, vy *= (1+ speed_increment)"""
        if self.bounce_count % self.speed_scale_every == 0:
            scale = 1.0 + self.speed_increment
            self.ball_vx *= scale
            self.ball_vy *= scale

    # ---------- 視角翻轉 -----------
    def _get_obs_for_A(self):
        # (A) => [ ball_x, (1-ball_y), ball_vx, (-ball_vy), top_paddle_x, bottom_paddle_x, spin ]
        return np.array([
            self.ball_x,
            1.0 - self.ball_y,
            self.ball_vx,
            -self.ball_vy,
            self.top_paddle_x,
            self.bottom_paddle_x,
            self.spin
        ], dtype=np.float32)

    def _get_obs_for_B(self):
        # (B) => [ ball_x, ball_y, ball_vx, ball_vy, bottom_paddle_x, top_paddle_x, spin ]
        return np.array([
            self.ball_x,
            self.ball_y,
            self.ball_vx,
            self.ball_vy,
            self.bottom_paddle_x,
            self.top_paddle_x,
            self.spin
        ], dtype=np.float32)

    def _get_obs(self):
        """回傳 (obsA, obsB) 共 7維"""
        obsA = self._get_obs_for_A()
        obsB = self._get_obs_for_B()
        return (obsA, obsB)

    def render(self):
        if not self.enable_render:
            return
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        self.screen.fill((0,0,0))

        # 畫球
        bx = int(self.ball_x * self.render_size)
        by = int(self.ball_y * self.render_size)
        r = 8  # 像素半徑
        pygame.draw.circle(self.screen, (255,255,255), (bx,by), r)

        # 繪製旋轉十字
        self.spin_angle += self.spin
        radius_cross = r - 2

        x1 = bx + radius_cross * math.cos(math.radians(self.spin_angle))
        y1 = by + radius_cross * math.sin(math.radians(self.spin_angle))
        x2 = bx - radius_cross * math.cos(math.radians(self.spin_angle))
        y2 = by - radius_cross * math.sin(math.radians(self.spin_angle))
        pygame.draw.line(self.screen, (255, 0, 0), (x1, y1), (x2, y2), 2)

        x3 = bx + radius_cross * math.cos(math.radians(self.spin_angle+90))
        y3 = by + radius_cross * math.sin(math.radians(self.spin_angle+90))
        x4 = bx - radius_cross * math.cos(math.radians(self.spin_angle+90))
        y4 = by - radius_cross * math.sin(math.radians(self.spin_angle+90))
        pygame.draw.line(self.screen, (255, 0, 0), (x3, y3), (x4, y4), 2)

        # 上板(A)
        tx = int(self.top_paddle_x*self.render_size)
        pw = int(self.paddle_width*self.render_size)
        pygame.draw.rect(self.screen, (0,255,0), (tx - pw//2, 0, pw, 10))

        # 下板(B)
        bx_ = int(self.bottom_paddle_x*self.render_size)
        pygame.draw.rect(self.screen, (0,255,0), (bx_ - pw//2, self.render_size -10, pw, 10))

        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        if self.enable_render:
            pygame.quit()
