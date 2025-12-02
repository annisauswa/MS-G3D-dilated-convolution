
def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

number_label = 52
theta = 0.5

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def aggregate_mapa(segments, num_classes):
    a_props = [[] for _ in range(num_classes)]
    for seg in segments:
        lab = int(seg[0])
        a_props[lab].append(list(seg)) 

    for c in range(num_classes):
        if a_props[c]:
            a_props[c].sort(key=lambda s: int(s[1]))
            
    return a_props

def calc_pr(positive, proposal, ground):
	if (proposal == 0): return 0,0
	if (ground == 0): return 0,0
	return (1.0*positive)/proposal, (1.0*positive)/ground

def match(lst, ratio, ground):
	def overlap(prop, ground):
		l_p, s_p, e_p, c_p = prop
		l_g, s_g, e_g = ground
		if (int(l_p) != int(l_g)): return 0
		if int(l_p) != int(l_g):
			return 0

		intersection = min(e_p, e_g) - max(s_p, s_g)
		union = max(e_p, e_g) - min(s_p, s_g)

		# prevent division by zero
		if union <= 0:
			return 0

		return max(0, intersection) / union
		# return (min(e_p, e_g)-max(s_p, s_g))/(max(e_p, e_g)-min(s_p, s_g))

	cos_map = [-1 for x in range(len(lst))]
	count_map = [0 for x in range(len(ground))]
	#generate index_map to speed up
	index_map = [[] for x in range(number_label)]
	for x in range(len(ground)):
		index_map[int(ground[x][0])].append(x)

	for x in range(len(lst)):
		for y in index_map[int(lst[x][0])]:
			if (overlap(lst[x], ground[y]) < ratio): continue
			if (overlap(lst[x], ground[y]) < overlap(lst[x], ground[cos_map[x]])): continue
			cos_map[x] = y
		if (cos_map[x] != -1): count_map[cos_map[x]] += 1
	positive = sum([(x>0) for x in count_map])
	return cos_map, count_map, positive

def f1(lst, ratio, ground):
	cos_map, count_map, positive = match(lst, ratio, ground)
	precision, recall = calc_pr(positive, len(lst), len(ground))

	if precision == 0 or recall == 0: return 0.0, positive, len(lst)-positive, len(ground)-positive
	
	score = 2*precision*recall/(precision+recall)
	return score, positive, len(lst)-positive, len(ground)-positive # F1, TP, FP, FN

def ap(lst, ratio, ground):
	lst.sort(key = lambda x:x[3]) # sorted by confidence
	cos_map, count_map, positive = match(lst, ratio, ground)
	score = 0
	number_proposal = len(lst)
	number_ground = len(ground)
	old_precision, old_recall = calc_pr(positive, number_proposal, number_ground)
 
	for x in range(len(lst)):
		number_proposal -= 1
		if (cos_map[x] == -1): continue
		count_map[cos_map[x]] -= 1
		if (count_map[cos_map[x]] == 0): positive -= 1

		precision, recall = calc_pr(positive, number_proposal, number_ground)   
		if precision>old_precision: 
			old_precision = precision
		score += old_precision*(old_recall-recall)
		old_recall = recall
	return score