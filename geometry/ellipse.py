import numpy as np

def least_squares_fit(x_coords, y_coords):
  # Algorithm adapted from:
  # http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.1.7559
  x_coords = np.asarray(x_coords)
  y_coords = np.asarray(y_coords)
  D1 = np.column_stack((x_coords**2, x_coords*y_coords, y_coords**2))
  D2 = np.column_stack((x_coords, y_coords, np.ones(len(x_coords))))
  S1 = D1.astype(np.double).T.dot(D1)
  S2 = D1.astype(np.double).T.dot(D2)
  S3 = D2.astype(np.double).T.dot(D2)
  T = -np.linalg.inv(S3).dot(S2.T)
  M = S1 + S2.dot(T) # reduced scatter matrix
  M = np.vstack([M[2] / 2.0, -M[1], M[0] / 2.0]) # premultiply by inv(C1)
  # M should be well-conditioned
  if np.linalg.cond(M) > 1e4:
    raise ValueError("Matrix of ellipse coordinates is ill-conditioned")
  evals, evecs = np.linalg.eig(M)
  cond = 4 * (evecs[0] * evecs[2]) - evecs[1]**2 # a^T . C . a
  a1 = evecs[:, np.where(cond > 0)[0][0]]
  return np.concatenate((a1, T.dot(a1)))

def general_to_standard(A, B, C, D, E, F):
  """
  Convert an ellipse in general form:
    Ax^2 + Bxy + Cy^2 + Dx + Ey + F = 0
  To standard form:
    ((x - x0) cos t - (y - y0) sin t)^2/a^2
    + ((x - x0) sin t + (y - y0) cos t)^2/b^2 = 1
  The ellipse has center (x0, y0), major and minor semi-axes a and b,
  and the angle to the semi-major axis is t.

  Parameters: A, B, C, D, E, F
  Returns: x0, y0, a, b, t
  """
  A, B, C, D, E, F = map(float, [A, B, C, D, E, F])
  # Matrix representation of conic section
  AQ = np.array([[A, B/2, D/2], [B/2, C, E/2], [D/2, E/2, F]])
  A33 = AQ[0:2, 0:2]
  # Formula for center
  x0, y0 = np.linalg.inv(A33).dot([-D/2, -E/2])
  # Each eigenvector of A33 lies along one of the axes
  evals, evecs = np.linalg.eigh(A33)
  # Semi-axes from reduced canonical equation
  a, b = np.sqrt(-np.linalg.det(AQ)/(np.linalg.det(A33)*evals))
  # Return major axis as "a" and angle to major axis between -pi/2 and pi/2
  t = np.arctan2(evecs[1,0], evecs[0,0]) # angle of axis "a"
  if b > a:
    a, b = b, a
    t += np.pi/2
  if   t < -np.pi/2: t += np.pi
  elif t >  np.pi/2: t -= np.pi
  return (x0, y0, a, b, t)
