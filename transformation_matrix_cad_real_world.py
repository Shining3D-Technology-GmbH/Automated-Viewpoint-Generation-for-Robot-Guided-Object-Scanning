import numpy as np
#import lambda
import ikpy
from math import atan2, asin, degrees
# আপনার দেওয়া প্রথম ট্রান্সফরমেশন ম্যাট্রিক্স (4x4 matrix)
# tf_cad_1_10 = np.array([[-0.1940934536312161, 0.4775535231433996, 0.8568957718362529, -313.0644503626668893],
#                      [0.9770022269270847, 0.0154901831689096, 0.2126657066969874, -118.9380068137469380],
#                      [0.0882857850225184, 0.8784660988101555, -0.4695773987366064, 209.8670278757941219],
#                      [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])

# # আপনার দেওয়া দ্বিতীয় ট্রান্সফরমেশন ম্যাট্রিক্স (4x4 matrix)
# tf_cad_2_10 = np.array([[-0.1931685448710397,0.4786292443257689, 0.8565045007170491, -312.7692974716336494],
#                      [0.9771437455708788, 0.0148862546408390, 0.2120577749446996, -118.7854153782567437],
#                      [0.0887469084764866, 0.8778909077436239, -0.4705651286877878, 210.3042532001842062],
#                      [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])
############view point 252 #####################
tf_cad_1_3 = np.array([[0.2447068433097801, 0.7931190286152381, -0.5577461495033343, 373.4470873976964640],
                        [0.9540108037172491, -0.0942227949656665, 0.2845794288061559, -135.6814163988356938],
                        [0.1731529590510737, -0.6017343860519291, -0.7797010846565351, 358.4657023550899453],
                        [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])


tf_cad_2_3= np.array([[0.2444279637097197, 0.7929267398402237, -0.5581416986779316, 365.5280705528484191],
                        [0.9542356448968367, -0.0944133721186099, 0.2837612538270938, -116.9351366957596809],
                        [0.1723058459978979, -0.6019578892344214, -0.7797162272412961, 358.7105881862560182],
                        [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])



#############view_point 239 #################
tf_cad_1_2 = np.array([[0.5252834455210043, -0.6186380630762591, 0.5842638520179483, -163.2355252216503914],
                        [-0.8329341415865568, -0.5142675024938689, 0.2043273150075491, -75.8392829328978877],
                        [0.1740632575848738, -0.5939830660818428, -0.7854209696506509, 359.1811059268285931],
                        [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])

# আপনার দেওয়া দ্বিতীয় ট্রান্সফরমেশন ম্যাট্রিক্স (4x4 matrix)
tf_cad_2_2= np.array([[0.5256126030693592, -0.6187593384549220, 0.5838392523370560, -152.6694637190089452],
                        [-0.8328049806848563, -0.5143575140479489, 0.2046270067437976, -79.4319308924522431],
                        [0.1736872351128964, -0.5937787709385165, -0.7856586507775456, 359.1148143362153746],
                        [0.0000000000000000, 0.0000000000000000, 0.0000000000000000, 1.0000000000000000]])

# আপনার প্রদত্ত প্রথম ট্রান্সফরমেশন ম্যাট্রিক্সটি ব্যবহার করা হচ্ছে
T = tf_cad_1_3

# রোটেশন অংশ বের করা (প্রথম তিনটি রো এবং কলাম)
R = T[:3, :3]

# আপনার প্রদত্ত ভেক্টর [0, 0, 1]
v = np.array([0, 0, 1])

# ভেক্টর এবং রোটেশন ম্যাট্রিক্সের গুনফল
rotated_v = np.dot(R, v)

# নতুন ট্রান্সফরমেশন ম্যাট্রিক্স তৈরি করা
new_T = np.eye(4)
new_T[:3, :3] = R  # রোটেশন অংশ অপরিবর্তিত থাকে
new_T[:3, 3] = T[:3, 3]  # ট্রান্সলেশন অংশ অপরিবর্তিত থাকে

# নতুন গুনফল ম্যাট্রিক্স প্রদর্শন করা
print("Rotated Vector 1:", rotated_v)
print("New Transformation Matrix :\n", new_T)
left_camera_position_cad_1 = new_T

T2 = tf_cad_2_3

# রোটেশন অংশ বের করা (প্রথম তিনটি রো এবং কলাম)
R2 = T2[:3, :3]

# আপনার প্রদত্ত ভেক্টর [0, 0, 1]
v2 = np.array([0, 0, 1])

# ভেক্টর এবং রোটেশন ম্যাট্রিক্সের গুনফল
rotated_v2 = np.dot(R2, v2)

# নতুন ট্রান্সফরমেশন ম্যাট্রিক্স তৈরি করা
new_T2 = np.eye(4)
new_T2[:3, :3] = R  # রোটেশন অংশ অপরিবর্তিত থাকে
new_T2[:3, 3] = T[:3, 3]  # ট্রান্সলেশন অংশ অপরিবর্তিত থাকে

# নতুন গুনফল ম্যাট্রিক্স প্রদর্শন করা
print("Rotated Vector 2:", rotated_v2)
print("New Transformation Matrix :\n", new_T2)
left_camera_position_cad_2 = new_T2

#######Transformation matrix generation LCp1_cad to LCp2_cad   LCp1_cad =T_cad_p1-p2 * LCp2_cad########

def calculate_transformation(T1, T2):
    """
    T1 এবং T2 ম্যাট্রিক্স দিয়ে tf ম্যাট্রিক্স গণনা করে
    T1 * tf = T2
    """
    # T1 এর ইনভার্স বের করা
    T1_inv = np.linalg.inv(T1)
    
    # tf ম্যাট্রিক্স গণনা করা
    tf = np.dot(T1_inv, T2)
    
    return tf

# আপনার দেওয়া প্রথম ট্রান্সফরমেশন ম্যাট্রিক্স (T1)
T1_cad_1 = new_T

# আপনার দেওয়া দ্বিতীয় ট্রান্সফরমেশন ম্যাট্রিক্স (T2)
T2_cad_2 = new_T2

# গণনা করা tf ম্যাট্রিক্স
T_cad_p1_p2 = calculate_transformation(T1_cad_1, T2_cad_2)

print("Transformation Matrix (tf):\n", T_cad_p1_p2)

# যাচাই করা: T1_cad_1 * tf_matrix = T2_cad_2
# T2_calculated = np.dot(T1_cad_1, T_cad_p1_p2)
# print("\nCalculated T2 (Should be close to T2_cad_2):\n", T2_calculated

############# T_Robot_Random_1_2 generation #####################

def euler_to_rotation_matrix(roll, pitch, yaw):

    # রোল (X-অক্ষের চারপাশে ঘূর্ণন)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(np.radians(roll)), -np.sin(np.radians(roll))],
                   [0, np.sin(np.radians(roll)), np.cos(np.radians(roll))]])

    # পিচ (Y-অক্ষের চারপাশে ঘূর্ণন)
    Ry = np.array([[np.cos(np.radians(pitch)), 0, np.sin(np.radians(pitch))],
                   [0, 1, 0],
                   [-np.sin(np.radians(pitch)), 0, np.cos(np.radians(pitch))]])

    # ইয়াউ (Z-অক্ষের চারপাশে ঘূর্ণন)
    Rz = np.array([[np.cos(np.radians(yaw)), -np.sin(np.radians(yaw)), 0],
                   [np.sin(np.radians(yaw)), np.cos(np.radians(yaw)), 0],
                   [0, 0, 1]])

    # মোট রোটেশন ম্যাট্রিক্স হলো R = Rz * Ry * Rx
    R = np.dot(Rz, np.dot(Ry, Rx))

    return R

def position_to_transformation_matrix(x, y, z, roll, pitch, yaw):
    """
    রোবট পজিশন থেকে 4x4 ট্রান্সফরমেশন ম্যাট্রিক্স তৈরি করা
    """
    # রোটেশন ম্যাট্রিক্স তৈরি করা
    R = euler_to_rotation_matrix(roll, pitch, yaw)

    # 4x4 ট্রান্সফরমেশন ম্যাট্রিক্স তৈরি করা
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]

    return T
def transformation_matrix_to_x_y_z_rx_ry_rz(matrix):
    # z y x
    x = matrix[0, 3]
    y = matrix[1, 3]
    z = matrix[2, 3]

    # Extract the rotation components
    r11, r12, r13 = matrix[0, :3]
    r21, r22, r23 = matrix[1, :3]
    r31, r32, r33 = matrix[2, :3]

    # Calculate pitch (ry)
    ry = asin(-r31)

    # Calculate roll (rx) and yaw (rz)
    if abs(r31) != 1:
        rx = atan2(r32 / np.cos(ry), r33 / np.cos(ry))
        rz = atan2(r21 / np.cos(ry), r11 / np.cos(ry))
    else:
        # Gimbal lock occurs, handle the singularity
        rz = 0
        if r31 == -1:
            ry = np.pi / 2
            rx = atan2(r12, r13)
        else:
            ry = -np.pi / 2
            rx = atan2(-r12, -r13)

    # Convert roll, pitch, yaw to degrees
    rx = degrees(rx)
    ry = degrees(ry)
    rz = degrees(rz)

    return x, y, z, rx, ry, rz

# প্রথম রোবট পজিশন
# position1 = [-408.057, 132.558, 456.612, 72.569, -21.806, -115.705]
# # দ্বিতীয় রোবট পজিশন
# position2 = [-441.738, 132.557, 456.612, 72.569, -21.806, -115.705]

#position1 = [-428.194,-38.552,370.459,154.003,-10.17,10.893]
#position1 = [-354.771,49.589,335.506,50.653,-18.495,-107.604]
#position1 = [-323.564,73.002,334.851,50.309,-18.418,-105.320]
position1 = [-354.771,49.589,335.56,50.653,-18.495,-107.604]
# দ্বিতীয় রোবট পজিশন
#position2 = [-427.993,-64.098,369.396,154.002,-10.819,16.881]
#position2 = [-354.789,60.702,335.505,50.663,-18.496,-107.604]
position2 = [-354.789,60.702,335.505,50.663,-18.496,-107.604]
# ট্রান্সফরমেশন ম্যাট্রিক্স তৈরি করা
T1 = position_to_transformation_matrix(*position1)
T2 = position_to_transformation_matrix(*position2)

print("Transformation Matrix T1:\n", T1)
print("\nTransformation Matrix T2:\n", T2)

# T1 এবং T2 এর মধ্যে ট্রান্সফরমেশন বের করা
T_Robot_Random_1_2 = np.dot(np.linalg.inv(T1), T2)
print("\nTransformation from Position 1 to Position 2:\n", T_Robot_Random_1_2)
##############_K_generation #####################
T_cad_robot =  np.dot(np.linalg.inv(T_cad_p1_p2), T_Robot_Random_1_2)
print("\nTransformation from CAD to Robot:\n", T_cad_robot)

################# T_p_82 generation #################
# target_position = np.array([-465.233, 115.672, 456.622])
# target_orientation_deg = np.array([72.858, -22.474, -115.466])
#best_view = [-428.193,-57.490,370.459,150.443,-10.717,16.893]
#best_view = [-326.277,49.588,335.504,50.563,-18.496,-107.604]
#best_view = [-323.564,47.909,334.853,50.309,-18.418,-107.504]
best_view = [-326.277,49.588,335.504,50.563,-18.196,-107.604]
#new_position= [-428.193,-57.490,370.459,150.443,-10.717,16.893]
new_position= [-354.771,49.589,335.56,50.653,-18.495,-107.604]
T3=position_to_transformation_matrix(*position1)
T4=position_to_transformation_matrix(*best_view)
T_p_82 = np.dot(np.linalg.inv(T3), T4)
print(T_p_82)
validation = np.dot( T3,T_p_82)
print(validation) ########ok here


########### T_p_82_robot_generation ###########
T_p_82_robot = np.dot(T_p_82, T_cad_robot)
print(T_p_82_robot)
new_robot=  np.dot(T3, T_p_82_robot)
print(new_robot)


# Convert matrix to position and orientation
x, y, z, rx, ry, rz = transformation_matrix_to_x_y_z_rx_ry_rz(new_robot)
print("x, y, z, rx, ry, rz :", (x, y, z, rx, ry, rz))
final_robot_position=np.array([x, y, z, rx, ry, rz])

def deviation_calculation(position_1,position_2):
    # position_1=T3
    # position_2=T4
    deviation=np.array(position_1) - np.array(position_2)
    print("Dispalcement between two random points in every direction :")
    print(deviation)
    return deviation


def add_displacement(position, displacement):

    position_array = np.array(position)
    displacement_array = np.array(displacement)
    #print(displacement_array)
    
    # Calculate the final position and orientation
    final_position = position_array + displacement_array
    
    return final_position
deviation = deviation_calculation(position1, position2)

position = [x, y, z, rx, ry, rz]

# Displacement vector (calculated previously)
# displacement_3 = position

# Calculate the final position
final_position = add_displacement(position, deviation)

print("Final position and orientation:", final_position)









