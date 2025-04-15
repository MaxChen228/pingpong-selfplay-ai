import math

def collide_sphere_with_moving_plane(vn, vt, u, omega, e, mu, m, R):
    """
    模擬球與移動平面碰撞，產生反彈、切向摩擦與角速度變化。
    """
    vn_post = - e * vn
    Jn = m * (1 + e) * abs(vn)
    I = (2/5) * m * R**2
    Jt_star = (2*m/7.0) * (u + R*omega - vt)
    max_friction_impulse = mu * Jn

    if abs(Jt_star) <= max_friction_impulse:
        Jt = Jt_star
    else:
        vrel = (vt - u) - R*omega
        sign_vrel = math.copysign(1, vrel)
        Jt = - max_friction_impulse * sign_vrel

    vt_post = vt + (Jt / m)
    omega_post = omega - (R * Jt) / I

    return vn_post, vt_post, omega_post