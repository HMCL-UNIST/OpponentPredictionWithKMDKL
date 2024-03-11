Header header
float32 t  # time in seconds

float32[] x  # global x coordinate in meters
float32[] y  # global y coordinate in meters

float32[] v_x  # global x velocity in m/s
float32[] v_y  # global y velocity in m/s

float32[] a_x  # global x acceleration in m/s^2
float32[] a_y  # global y acceleration in m/s^2

float32[] psi  # global vehicle heading angle
float32[] psidot  # global and local angular velocity of car

float32[] v_long  # longitudinal velocity (in the direction of psi)
float32[] v_tran  # transverse velocity (orthogonal to the direction of psi)

float32[] a_long  # longitudinal acceleration (in the direction of psi)
float32[] a_tran  # transverse acceleration (orthogonal to the direction of psi)

float32[] e_psi  # heading error between car and track
float32[] s  # path length along center of track to projected position of car
float32[] x_tran  # deviation from centerline (transverse position)

float32[] u_a  # acceleration output
float32[] u_steer  # steering angle output

int32 lap_num

float32[] sey_cov  # covariance matrix in sey frame

float32[] xy_cov  # covariance matrix in local heading frame