import redis

import json

import logging

import config



logger = logging.getLogger(__name__)



# Use a connection pool for efficiency
# Uses config values which support environment variables

try:
    # Create connection pool with authentication support
    pool_kwargs = {
        'host': config.REDIS_HOST,
        'port': config.REDIS_PORT,
        'db': config.REDIS_DB,
        'decode_responses': True
    }

    # Add authentication if password is provided
    if config.REDIS_PASSWORD:
        pool_kwargs['password'] = config.REDIS_PASSWORD
        pool_kwargs['username'] = config.REDIS_USERNAME

    pool = redis.ConnectionPool(**pool_kwargs)
    logger.info(f"Redis pool created: {config.REDIS_HOST}:{config.REDIS_PORT} (auth: {'yes' if config.REDIS_PASSWORD else 'no'})")

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

    # Only send progress updates if enabled, always send complete/error
    if not config.SEND_PROGRESS_UPDATES and status not in ["complete", "error"]:
        logger.debug(f"Skipping progress update for {job_id}: {status} - {detail}")
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

