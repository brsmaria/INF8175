import random
from collections import defaultdict

def solve(schedule):
    """
    Your solution of the problem
    :param schedule: object describing the input
    :return: a list of tuples of the form (c,t) where c is a course and t a time slot. 
    """
    solution = generateInitialSolution(schedule)
    current_cost = evaluateSolution(solution, schedule)
    no_improvement_count = 0

    while no_improvement_count < 50:
        neighbour_solution = getNeighbour(solution)
        neighbour_cost = evaluateSolution(neighbour_solution, schedule)

        if neighbour_cost < current_cost:
            solution = neighbour_solution
            current_cost = neighbour_cost
            no_improvement_count = 0
        else:
            no_improvement_count += 1 
    
    return solution

def generateInitialSolution(schedule):
    solution = dict()
    incompatibles_courses_per_time_slot = defaultdict(set) # Mappe un créneau avec les cours qui ne peuvent pas avoir lieu pendant celui-ci

    sorted_courses = sorted(schedule.course_list, key=lambda c: len(schedule.get_node_conflicts(c)), reverse=True)

    for course in sorted_courses:
        assigned = False
        for time_slot_idx in range(1, len(schedule.course_list) + 1): # Au cas où il n'y a que des conflits et qu'on doit avoir autant de créneau que de cours
            if course not in incompatibles_courses_per_time_slot[time_slot_idx]:
                solution[course] = time_slot_idx
                incompatibles_courses_per_time_slot[time_slot_idx].update(schedule.get_node_conflicts(course))
                assigned = True
                break

        if not assigned:
            new_time_slot_idx = time_slot_idx + 1
            solution[course] = new_time_slot_idx
            incompatibles_courses_per_time_slot[new_time_slot_idx].update(schedule.get_node_conflicts(course))

    return solution

def getNeighbour(solution):
    neighbour = solution.copy()
    course_to_move = random.choice(list(neighbour.keys()))
    used_slots = set(neighbour.values())
    
    new_slot = random.choice([slot for slot in used_slots if slot != neighbour[course_to_move]])
    neighbour[course_to_move] = new_slot
    
    return neighbour

def evaluateSolution(solution, schedule):
    conflicts = set()
    time_slots = defaultdict(set)

    for course, time_slot in solution.items():
        for conflicting_course in schedule.get_node_conflicts(course):
            if conflicting_course in time_slots[time_slot]:
                pair = (course, conflicting_course)
                conflicts.add(tuple(sorted(pair)))
        time_slots[time_slot].add(course)

    return conflicts # Solution valide si conflicts est vide sinon invalide