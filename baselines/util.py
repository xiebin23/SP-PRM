import time


def timeit(func):
    def wrapper(*args, **kwargs):
        import time
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        elapsed = end - start
        return result, elapsed
    return wrapper


def apply_chat_template(conversation):

    return "\n\n".join(
        f"{'Human' if qa['role'] == 'user' else 'Assistant'}: {qa['content']}" for qa in conversation)
