#Functions used to calculate epsilon-BMC

class Beta:
    # parameters of beta
    def __init__(self, alpha0, beta0):
        self.alpha = [alpha0]
        self.beta = [beta0]

    def update(self, expert1, expert2):
        """
        Update alpha and beta with Dirichlet moment-matching technique
        """
        list_length = len(self.alpha)
        alpha, beta = self.alpha[list_length-1], self.beta[list_length-1]
        mean = expert1 * alpha + expert2 * beta
        if mean <= 0.0:
            return
        m = alpha / (alpha + beta + 1.) * (expert1 * (alpha + 1.) + expert2 * beta) / mean
        s = alpha / (alpha + beta + 1.) * (alpha + 1.) / (alpha + beta + 2.) * \
            (expert1 * (alpha + 2.) + expert2 * beta) / mean
        r = (m - s) / (s - m * m)
        self.alpha.append(m * r)
        self.beta.append((1. - m) * r)


class Average:

    def __init__(self, mean=0.0, m2=0.0, count=0):
        self.mean, self.m2, self.count = mean, m2, count
        self.var = [0.0]

    def update(self, point):
        """
        Update mean and variance of variance of returns
        """
        self.count += 1
        count = self.count
        delta = point - self.mean
        self.mean += delta / count
        self.m2 += delta * (point - self.mean)
        if count > 1:
            self.var.append(self.m2 / (count - 1.0))
