import redis

import json

import logging



logger = logging.getLogger(__name__)



# Use a connection pool for efficiency

# Ensure your Redis server is running on localhost:6379

try:

    pool = redis.ConnectionPool(host='localhost', port=6379, db=0, decode_responses=True)

except Exception as e:

    logger.error(f"Could not create Redis connection pool: {e}")

    pool = None



def publish_update(job_id: str, status: str, result: dict = None, error: str = None, detail: str = None):

    """

    Publishes a JSON message to the job's specific Redis channel.

    Hardcoded for Rainbow Six Siege (r6s).

    """

    if pool is None:

        logger.error("Redis pool is not available. Cannot publish update.")

        return



    try:

        r = redis.Redis(connection_pool=pool)

        

        # --- HARDCODED CHANNEL ---

        # This ensures all messages automatically go to the 'r6s' namespace

        channel = f"job_updates:r6s:{job_id}"

        

        message = {

            "job_id": job_id,

            "status": status,

            "result": result,

            "error": error,

            "detail": detail,

            "game": "r6s" # Included in payload just in case frontend needs it

        }

        

        # Publish to Redis

        r.publish(channel, json.dumps(message))

        

    except Exception as e:

        # Log error but don't crash the analysis pipeline

        logger.error(f"Error publishing to Redis channel {channel}: {e}")